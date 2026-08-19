"""
Jira weekly status update tool.

Calls gather_status_data.py + summarize_issue.py as subprocesses (unchanged),
drafts R/Y/G status updates via the Anthropic API (all issues in parallel),
then runs an interactive review loop (CLI or Textual TUI) that writes approved
drafts to Jira customfield_10814.
"""

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp
import anthropic

import tomllib

import pyadf

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, ProgressBar, RichLog, Static


# ─── Constants ───────────────────────────────────────────────────────────────

LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
_pricing_cache: Optional[dict] = None

_SKILL_DIR_REL = Path("plugins/jira/skills/status-analysis")


# ─── UserPrefs ───────────────────────────────────────────────────────────────

@dataclass
class UserPrefs:
    model: str = "claude-sonnet-5"
    coding_agent: Optional[str] = None
    writing_rules: list = field(default_factory=list)
    jira_url: str = "https://redhat.atlassian.net"
    skills_dir: Optional[str] = None
    profiles: dict = field(default_factory=dict)


def load_prefs(path: Optional[Path] = None) -> UserPrefs:
    """Load from ~/.config/jira-status-update/prefs.toml or the given path."""
    if path is None:
        xdg = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        path = Path(xdg) / "jira-status-update" / "prefs.toml"
    if not path.exists():
        return UserPrefs()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    llm = data.get("llm", {})
    jira = data.get("jira", {})
    return UserPrefs(
        model=llm.get("model", "claude-sonnet-5"),
        coding_agent=llm.get("coding_agent"),
        writing_rules=llm.get("writing_rules", []),
        jira_url=jira.get("url", "https://redhat.atlassian.net"),
        skills_dir=jira.get("skills_dir"),
        profiles=data.get("profiles", {}),
    )


# ─── LLM Client ──────────────────────────────────────────────────────────────

def create_client():
    """Create Anthropic client — Vertex AI if configured, else direct API."""
    vertex_project = (
        os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    vertex_region = os.environ.get("CLOUD_ML_REGION", "us-east5")
    if vertex_project:
        return anthropic.AsyncAnthropicVertex(
            project_id=vertex_project, region=vertex_region
        )
    return anthropic.AsyncAnthropic()


# ─── Cost Tracking ───────────────────────────────────────────────────────────

def fetch_model_pricing(model_id: str, is_vertex: bool = False) -> Optional[dict]:
    """Fetch per-token pricing from the LiteLLM database with module-level cache."""
    global _pricing_cache
    if _pricing_cache is None:
        try:
            resp = urllib.request.urlopen(LITELLM_PRICING_URL, timeout=5)
            _pricing_cache = json.loads(resp.read().decode())
        except Exception as e:
            print(f"Warning: Failed to fetch model pricing: {e}", file=sys.stderr)
            return None

    lookup = model_id
    if is_vertex:
        lookup = f"vertex_ai/{model_id}"

    raw = _pricing_cache.get(lookup)
    if raw is None:
        candidates = [k for k in _pricing_cache if model_id in k]
        if candidates:
            raw = _pricing_cache[candidates[0]]
    if raw is None:
        print(f"Warning: Model '{model_id}' not found in pricing database", file=sys.stderr)
        return None

    return {
        "input": raw.get("input_cost_per_token", 0) * 1_000_000,
        "output": raw.get("output_cost_per_token", 0) * 1_000_000,
        "cache_write": raw.get("cache_creation_input_token_cost", 0) * 1_000_000,
        "cache_read": raw.get("cache_read_input_token_cost", 0) * 1_000_000,
    }


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    def add(self, response) -> None:
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.cache_write_tokens += getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(response.usage, "cache_read_input_tokens", 0) or 0
        self.calls += 1

    def cost(self, pricing: dict) -> float:
        return (
            self.input_tokens * pricing["input"] / 1_000_000
            + self.output_tokens * pricing["output"] / 1_000_000
            + self.cache_write_tokens * pricing.get("cache_write", 0) / 1_000_000
            + self.cache_read_tokens * pricing.get("cache_read", 0) / 1_000_000
        )


# ─── Core LLM Functions ──────────────────────────────────────────────────────

def _first_text(response) -> str:
    """Return the text of the first TextBlock, skipping any ThinkingBlocks."""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    for block in response.content:
        if hasattr(block, "thinking"):
            return block.thinking
    raise ValueError(f"No usable content in response: {response.content}")


async def draft_ryg(
    client, model: str, system_prompt: str, issue_summary: str, usage: TokenUsage
) -> str:
    """Draft R/Y/G status for one issue. One independent API call."""
    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": issue_summary}],
    )
    usage.add(response)
    return _first_text(response)


def _resolve_coding_agent(prefs: UserPrefs) -> Optional[str]:
    """Return the coding agent CLI path from prefs, or None if unset/missing."""
    import shutil
    name = prefs.coding_agent
    if not name:
        return None
    return shutil.which(name)


async def request_changes_api(
    client, model: str, draft: str, issue_summary: str, instruction: str, usage: TokenUsage,
) -> str:
    """Revise a draft via a single Anthropic API call (no tool access)."""
    return await _request_changes_api(client, model, draft, issue_summary, instruction, usage)


def _build_agent_prompt(
    draft: str, issue_summary: str, instruction: str, tmppath: str,
    issue_key: Optional[str] = None, date_dir: Optional[Path] = None,
) -> str:
    """Build the prompt passed to the coding agent CLI."""
    data_context = ""
    if date_dir and issue_key:
        issue_json = date_dir / "issues" / f"{issue_key}.json"
        data_context = (
            f"Gathered Jira and GitHub data for this issue is at {issue_json}. "
            f"The full data directory is {date_dir}/issues/ — read the JSON "
            f"file to see PRs, changelog, comments, and descendant issues.\n"
        )
    return (
        f"I need you to revise a Jira R/Y/G status update.\n\n"
        f"The current draft is in {tmppath}.\n\n"
        f"Issue context:\n{issue_summary}\n\n"
        f"{data_context}\n"
        f"Requested change: {instruction}\n\n"
        f"Instructions:\n"
        f"1. Read {tmppath} to see the current draft.\n"
        f"2. If the request involves PRs, commits, or code changes, read the "
        f"gathered data JSON above and/or use gh CLI to search GitHub.\n"
        f"3. Write the complete revised status update to {tmppath}, replacing "
        f"its entire contents. The output must keep the exact same format — "
        f"starting with '* Color Status:'. No preamble, no explanation, just "
        f"the status update.\n"
    )


def _run_agent_interactive(
    agent: str, model: str, draft: str, issue_summary: str, instruction: str,
    issue_key: Optional[str] = None, date_dir: Optional[Path] = None,
) -> str:
    """Run the coding agent interactively (blocking, owns the terminal).

    Called inside app.suspend() so the agent gets full terminal control.
    """
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(draft)
        draft_path = f.name
    prompt = _build_agent_prompt(
        draft, issue_summary, instruction, draft_path,
        issue_key=issue_key, date_dir=date_dir,
    )
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(prompt)
        prompt_path = f.name
    short_prompt = f"Read your full instructions from {prompt_path} and follow them."
    agent_name = os.path.basename(agent)
    if agent_name == "opencode":
        cmd = [agent, "run", "-i", "--model", model, short_prompt]
    else:
        cmd = [agent, "--model", model, short_prompt]
    import subprocess as sp
    try:
        sp.run(cmd)
    finally:
        os.unlink(prompt_path)
    try:
        with open(draft_path) as f:
            result = f.read().strip()
    finally:
        os.unlink(draft_path)
    return result if result else draft


async def _request_changes_api(
    client, model: str, draft: str, issue_summary: str, instruction: str, usage: TokenUsage,
) -> str:
    """Fallback: revise via a single Anthropic API call (no tool access)."""
    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        system="You are revising a Jira R/Y/G status update. Output ONLY the revised status update in the exact same format.",
        messages=[{
            "role": "user",
            "content": (
                f"Issue summary:\n{issue_summary}\n\n"
                f"Current draft:\n{draft}\n\n"
                f"Requested change: {instruction}"
            ),
        }],
    )
    usage.add(response)
    return _first_text(response)


# ─── Cache ───────────────────────────────────────────────────────────────────

def _get_cache_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", os.environ.get("APPDATA", os.path.expanduser("~")))
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(base) / "jira-status-update"


def _cache_key(args) -> str:
    """Deterministic cache key from the gather parameters."""
    import hashlib
    parts = [
        args.project,
        getattr(args, "component", None) or "",
        getattr(args, "label", None) or "",
        ",".join(sorted(getattr(args, "assignee", None) or [])),
        ",".join(sorted(getattr(args, "exclude", None) or [])),
        str(getattr(args, "days", 7)),
        getattr(args, "start_date", None) or "",
        getattr(args, "end_date", None) or "",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _find_cached_gather(args) -> Optional[tuple]:
    """Return (date_dir, manifest) from cache, or None if stale/missing.

    Cache expires when the meta file was written on a different calendar day
    (UTC) than today, since the default gather range ends "today".
    """
    from datetime import datetime, timezone
    cache_dir = _get_cache_dir() / _cache_key(args)
    if not cache_dir.exists():
        return None
    meta_path = cache_dir / "cache_meta.json"
    if not meta_path.exists():
        return None
    mtime = datetime.fromtimestamp(meta_path.stat().st_mtime, tz=timezone.utc)
    if mtime.date() != datetime.now(tz=timezone.utc).date():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        date_dir = Path(meta["date_dir"])
        manifest_path = date_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path) as f:
            manifest = json.load(f)
        return date_dir, manifest
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _save_cache_meta(args, date_dir: Path) -> None:
    cache_dir = _get_cache_dir() / _cache_key(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "cache_meta.json", "w") as f:
        json.dump({"date_dir": str(date_dir)}, f)


# ─── Path Resolution ─────────────────────────────────────────────────────────

_AI_HELPERS_REPO = "https://github.com/openshift-eng/ai-helpers"


def _get_data_dir() -> Path:
    """Return the platform-appropriate user data directory for this app.

    - Linux:   $XDG_DATA_HOME/jira-status-update/  (default ~/.local/share/)
    - macOS:   ~/Library/Application Support/jira-status-update/
    - Windows: %APPDATA%\\jira-status-update\\
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return Path(base) / "jira-status-update"


def _fetch_ai_helpers(data_dir: Path) -> Path:
    """Clone ai-helpers into data_dir and return the resolved skills dir."""
    import shutil
    if not shutil.which("git"):
        raise RuntimeError(
            "Skills directory not found and 'git' is unavailable to fetch it.\n"
            "Install git, or set [jira] skills_dir in prefs.toml, or set\n"
            "JIRA_STATUS_UPDATE_SKILLS_DIR."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = data_dir / "ai-helpers"
    print(f"Fetching ai-helpers to {repo_dir} ...", file=sys.stderr)
    subprocess.run(
        ["git", "clone", "--depth=1", _AI_HELPERS_REPO, str(repo_dir)],
        check=True,
    )
    skills_dir = repo_dir / _SKILL_DIR_REL
    if not skills_dir.exists():
        raise FileNotFoundError(f"Expected skills dir not found after clone: {skills_dir}")
    return skills_dir


def find_skills_dir(prefs: Optional["UserPrefs"] = None) -> Path:
    """Resolve the status-analysis skills directory using a four-step search.

    1. [jira] skills_dir in prefs.toml (explicit override)
    2. JIRA_STATUS_UPDATE_SKILLS_DIR env var
    3. Platform data dir — previously auto-fetched copy:
         Linux:   ~/.local/share/jira-status-update/ai-helpers/
         macOS:   ~/Library/Application Support/jira-status-update/ai-helpers/
    4. Repo-relative path (when running directly from the source tree)
    5. Auto git-clone of openshift-eng/ai-helpers into the data directory
    """
    if prefs and prefs.skills_dir:
        p = Path(prefs.skills_dir).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"skills_dir from prefs not found: {p}")

    env = os.environ.get("JIRA_STATUS_UPDATE_SKILLS_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p
        raise FileNotFoundError(f"JIRA_STATUS_UPDATE_SKILLS_DIR not found: {p}")

    cached = _get_data_dir() / "ai-helpers" / _SKILL_DIR_REL
    if cached.exists():
        return cached

    dev = Path(__file__).resolve().parent.parent.parent / _SKILL_DIR_REL
    if dev.exists():
        return dev

    return _fetch_ai_helpers(_get_data_dir())


def find_scripts_dir(prefs: Optional["UserPrefs"] = None) -> Path:
    """Return the scripts/ subdirectory of the resolved skills dir."""
    return find_skills_dir(prefs) / "scripts"


# ─── Data Gathering ──────────────────────────────────────────────────────────

async def run_gather(
    scripts_dir: Path,
    args,
    output_dir: str = ".work/weekly-status",
    line_callback=None,
    refresh: bool = False,
    issue_keys: Optional[list[str]] = None,
) -> tuple:
    """Run gather_status_data.py and return (date_dir, manifest).

    Uses a cache keyed on the gather parameters. Pass refresh=True to force
    re-gather. The script writes manifest JSON to stdout and logs to stderr.
    line_callback, if provided, is called with each stderr line as it arrives.
    date_dir is reconstructed from the manifest's date_range.end field.
    When issue_keys is provided, only those specific issues are gathered.
    """
    if not refresh:
        cached = _find_cached_gather(args)
        if cached:
            if line_callback:
                await line_callback(f"Using cached data from {cached[0]}")
            return cached
    cmd = [
        sys.executable,
        str(scripts_dir / "gather_status_data.py"),
        "--project", args.project,
        "--output-dir", output_dir,
        "--verbose",
    ]
    if getattr(args, "component", None):
        cmd.extend(["--component", args.component])
    if getattr(args, "label", None):
        cmd.extend(["--label", args.label])
    for assignee in (getattr(args, "assignee", None) or []):
        cmd.extend(["--assignee", assignee])
    for excluded in (getattr(args, "exclude", None) or []):
        cmd.extend(["--exclude-assignee", excluded])
    if getattr(args, "days", None):
        cmd.extend(["--days", str(args.days)])
    if getattr(args, "start_date", None):
        cmd.extend(["--start-date", args.start_date])
    if getattr(args, "end_date", None):
        cmd.extend(["--end-date", args.end_date])
    for key in (issue_keys or []):
        cmd.extend(["--issue", key])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stderr_lines: list = []

    async def _drain_stderr():
        async for raw in proc.stderr:
            line = raw.decode().rstrip()
            stderr_lines.append(line)
            if line_callback:
                await line_callback(line)

    stdout_bytes, _ = await asyncio.gather(proc.stdout.read(), _drain_stderr())
    await proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("gather_status_data.py failed:\n" + "\n".join(stderr_lines))

    new_manifest = json.loads(stdout_bytes.decode())
    end_date = new_manifest["config"]["date_range"]["end"]
    date_dir = Path(output_dir) / end_date

    if issue_keys:
        manifest_path = date_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                existing = json.load(f)
            refreshed = {i["key"]: i for i in new_manifest.get("issues", [])}
            merged = [refreshed.pop(i["key"], i) for i in existing["issues"]]
            merged.extend(refreshed.values())
            existing["issues"] = merged
            with open(manifest_path, "w") as f:
                json.dump(existing, f)
            return date_dir, existing
        return date_dir, new_manifest

    _save_cache_meta(args, date_dir)
    return date_dir, new_manifest


# ─── Issue JSON Processing ───────────────────────────────────────────────────

def _field_to_text(value) -> str:
    """Convert a field value to plain text, handling ADF dicts."""
    if not value:
        return ""
    if isinstance(value, dict):
        try:
            return pyadf.Document(value).to_markdown()
        except Exception:
            return json.dumps(value)
    return str(value)


def is_significant(data: dict) -> bool:
    """Return True if the issue had meaningful activity in the reporting range."""
    if bool(data.get("prs")):
        return True
    for cl in data.get("changelog_in_range", []):
        if any(i.get("field") == "status" for i in cl.get("items", [])):
            return True
    if any(not c.get("is_bot") for c in data.get("comments_in_range", [])):
        return True
    desc = data.get("descendants", {})
    if isinstance(desc, dict) and desc.get("updated_in_range"):
        return True
    for pr in data.get("prs", []):
        if pr.get("activity_summary", {}).get("commits_in_range"):
            return True
        if pr.get("activity_summary", {}).get("reviews_in_range"):
            return True
    return False


def _prune_issue(data: dict) -> dict:
    """Return a compact subset of the per-issue JSON suitable for LLM input."""
    issue = data.get("issue", data)
    assignee = issue.get("assignee")
    pruned = {
        "key": issue.get("key"),
        "summary": issue.get("summary"),
        "status": issue.get("status"),
        "assignee": assignee.get("name", "?") if isinstance(assignee, dict) else (assignee or "?"),
        "current_status_summary": _field_to_text(issue.get("current_status_summary") or ""),
        "last_status_update": issue.get("last_status_summary_update", ""),
    }

    desc = data.get("descendants", {})
    if isinstance(desc, dict):
        pruned["descendants"] = {
            "total": desc.get("total", 0),
            "completion_pct": round(desc.get("completion_pct", 0.0)),
            "by_status": desc.get("by_status", {}),
            "updated_in_range": desc.get("updated_in_range", []),
        }

    pruned["changelog"] = [
        {
            "date": i.get("date"),
            "author": i.get("author"),
            "changes": [
                {"field": c.get("field"), "from": c.get("from"), "to": c.get("to")}
                for c in i.get("items", [])
            ],
        }
        for i in data.get("changelog_in_range", [])
    ]

    pruned["comments"] = [
        {
            "author_name": c.get("author_name"),
            "body": c.get("body"),
        }
        for c in data.get("comments_in_range", [])
        if not c.get("is_bot")
    ]

    pruned["prs"] = [
        {
            "number": pr.get("number"),
            "title": pr.get("title"),
            "state": pr.get("state"),
            "is_draft": pr.get("is_draft", False),
            "merged_at": pr.get("merged_at"),
            "dates": pr.get("dates", {}),
            "commits_in_range": pr.get("activity_summary", {}).get("commits_in_range", 0),
            "reviews_in_range": pr.get("activity_summary", {}).get("reviews_in_range", 0),
        }
        for pr in data.get("prs", [])
    ]

    return pruned


def load_current_statuses(date_dir: Path, issue_keys: list) -> dict:
    """Read the previous week's filed status summary from each per-issue JSON file."""
    statuses: dict[str, str] = {}
    for key in issue_keys:
        path = date_dir / "issues" / f"{key}.json"
        try:
            with open(path) as f:
                data = json.load(f)
            statuses[key] = _field_to_text(data["issue"].get("current_status_summary") or "")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            statuses[key] = ""
    return statuses


def _drafts_path(date_dir: Path) -> Path:
    return date_dir / "drafts.json"


def load_saved_drafts(date_dir: Path) -> dict:
    """Load previously saved drafts from disk, or empty dict."""
    path = _drafts_path(date_dir)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_drafts(date_dir: Path, drafts: dict) -> None:
    """Persist current drafts to disk."""
    path = _drafts_path(date_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(drafts, f, indent=2)


# ─── Batch Drafting ──────────────────────────────────────────────────────────

async def batch_draft(
    client,
    model: str,
    system_prompt: str,
    date_dir: Path,
    issues: list,
    usage: TokenUsage,
    callback=None,
    only_significant: bool = False,
) -> tuple:
    """Draft R/Y/G for all issues in parallel; return (drafts, summaries, failed) tuple."""
    saved = load_saved_drafts(date_dir)
    sem = asyncio.Semaphore(10)
    max_retries = 3

    async def draft_one(issue, idx):
        async with sem:
            key = issue["key"]
            if key in saved:
                if callback:
                    await callback(idx + 1, len(issues), f"Loaded {key} (cached draft)")
                return key, saved[key], "(cached)"

            path = date_dir / "issues" / f"{key}.json"
            try:
                with open(path) as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                if callback:
                    await callback(idx + 1, len(issues), f"Skipped {key} (no data file)")
                return key, None, None

            if only_significant and not is_significant(data):
                if callback:
                    await callback(idx + 1, len(issues), f"Skipped {key} (not significant)")
                return key, None, None

            try:
                summary = json.dumps(_prune_issue(data), indent=2)
            except Exception as e:
                msg = f"Failed {key} (prune): {e}"
                print(f"Warning: {msg}", file=sys.stderr)
                if callback:
                    await callback(idx + 1, len(issues), msg)
                return key, None, None

            last_err = None
            for attempt in range(max_retries):
                try:
                    draft = await draft_ryg(client, model, system_prompt, summary, usage)
                    if callback:
                        await callback(idx + 1, len(issues), f"Drafted {key}")
                    return key, draft, summary
                except Exception as e:
                    last_err = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)

            msg = f"Failed {key} (draft, {max_retries} attempts): {last_err}"
            print(f"Warning: {msg}", file=sys.stderr)
            if callback:
                await callback(idx + 1, len(issues), msg)
            return key, None, None

    tasks = [draft_one(issue, i) for i, issue in enumerate(issues)]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    drafts: dict[str, str] = {}
    result_summaries: dict[str, str] = {}
    failed: list[str] = []
    for r in raw_results:
        if isinstance(r, Exception):
            print(f"Warning: draft failed: {r}", file=sys.stderr)
            continue
        key, draft, summary = r
        if draft:
            drafts[key] = draft
        elif key:
            failed.append(key)
        if summary and summary != "(cached)":
            result_summaries[key] = summary

    save_drafts(date_dir, drafts)
    return drafts, result_summaries, failed


# ─── Jira Update ─────────────────────────────────────────────────────────────

def _text_to_adf(text: str) -> dict:
    """Convert markdown text into an ADF document for Jira Cloud API v3."""
    return pyadf.Document.from_markdown(text).to_adf()


@dataclass
class JiraUpdateResult:
    ok: bool
    error: Optional[str] = None


async def update_jira_status(
    session: aiohttp.ClientSession,
    jira_url: str,
    auth_headers: dict,
    issue_key: str,
    status_text: str,
) -> JiraUpdateResult:
    """Set customfield_10814 on a Jira issue.

    After a successful PUT, reads the field back to verify it actually changed.
    """
    url = f"{jira_url}/rest/api/3/issue/{issue_key}"
    payload = {"fields": {"customfield_10814": _text_to_adf(status_text)}}
    async with session.put(url, json=payload, headers=auth_headers) as resp:
        if resp.status not in (200, 204):
            text = await resp.text()
            if resp.status == 403:
                return JiraUpdateResult(
                    False,
                    f"No permission to update {issue_key} (customfield_10814). "
                    f"Check your API token has edit access.",
                )
            return JiraUpdateResult(
                False, f"Failed to update {issue_key}: {resp.status} {text}"
            )

    verify_url = f"{jira_url}/rest/api/3/issue/{issue_key}?fields=customfield_10814"
    async with session.get(verify_url, headers=auth_headers) as resp:
        if resp.status not in (200, 204):
            return JiraUpdateResult(
                False,
                f"PUT for {issue_key} returned 204 but verify GET failed "
                f"({resp.status}). Update may not have persisted.",
            )
        data = await resp.json()
        actual = data.get("fields", {}).get("customfield_10814")
        if actual is None:
            return JiraUpdateResult(
                False,
                f"{issue_key} customfield_10814 is null after PUT. The field "
                f"may not exist on this issue type or was silently rejected.",
            )

    return JiraUpdateResult(True)


def get_jira_auth() -> dict:
    """Build Jira auth headers from JIRA_USERNAME + JIRA_API_TOKEN env vars."""
    username = os.environ.get("JIRA_USERNAME", "")
    token = os.environ.get("JIRA_API_TOKEN", "")
    if not username or not token:
        raise ValueError("JIRA_USERNAME and JIRA_API_TOKEN must be set")
    credentials = base64.b64encode(f"{username}:{token}".encode()).decode()
    return {
        "Authorization": f"Basic {credentials}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


# ─── Editor Integration ───────────────────────────────────────────────────────

def edit_in_editor(draft_text: str) -> Optional[str]:
    """Open $EDITOR with draft text. Returns edited text or None if cancelled."""
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(draft_text)
        f.write(
            "\n\n# ---- Edit above this line. Lines starting with # are ignored. ----\n"
            "# Save and quit to apply. Delete all content and quit to cancel.\n"
        )
        tmppath = f.name

    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))
    try:
        subprocess.run([editor, tmppath], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Editor failed: {e}", file=sys.stderr)
        os.unlink(tmppath)
        return None

    with open(tmppath) as f:
        lines = [line for line in f.readlines() if not line.startswith("#")]
    os.unlink(tmppath)
    result = "".join(lines).strip()
    return result if result else None


# ─── Color Extraction ─────────────────────────────────────────────────────────

def extract_color(draft: str) -> str:
    """Extract R/Y/G color from a draft status update. Defaults to Green."""
    for line in draft.split("\n"):
        if "Color Status:" in line:
            for color in ("Red", "Yellow", "Green"):
                if color in line:
                    return color
    return "Green"


# ─── Review Result ───────────────────────────────────────────────────────────

@dataclass
class ReviewResult:
    key: str
    action: str   # "approved", "skipped", "edited", "revised"
    color: str    # "Green", "Yellow", "Red"
    text: str


# ─── System Prompt Builder ───────────────────────────────────────────────────

def build_system_prompt(prefs: UserPrefs) -> str:
    """Build system prompt from the canonical skill documents.

    activity-analysis.md and formatting.md are the single source of truth for
    R/Y/G rules. Including them verbatim here means the app stays in sync with
    the Claude Code skill automatically. The prompt is identical across all
    parallel API calls, so Anthropic's prompt caching applies — tokens after
    the first call cost ~10% of normal input rates.
    """
    skills_dir = find_skills_dir(prefs)
    parts = [
        "You are a Jira status analyst producing weekly R/Y/G status updates.",
        "Apply the analysis methodology and output format defined below.\n",
    ]
    for filename in ("activity-analysis.md", "formatting.md"):
        path = skills_dir / filename
        try:
            parts.append(f"## {filename}\n\n{path.read_text()}\n")
        except FileNotFoundError:
            pass

    if prefs.writing_rules:
        rules = "\n".join(f"- {rule}" for rule in prefs.writing_rules)
        parts.append(f"## Additional writing rules (user preferences)\n\n{rules}\n")

    parts.append(
        "Produce only the `ryg_field` format output defined in formatting.md. "
        "No preamble, no explanation — output starts with `* Color Status:`."
    )
    return "\n".join(parts)


# ─── Headless Mode ───────────────────────────────────────────────────────────

async def run_headless(args, prefs: UserPrefs, scripts_dir: Path) -> dict:
    """Gather + draft, return JSON dict (no interactive review)."""
    refresh = getattr(args, "refresh", False)
    date_dir, manifest = await run_gather(scripts_dir, args, args.output_dir, refresh=refresh)
    client = create_client()
    model = args.model or prefs.model
    usage = TokenUsage()
    system_prompt = build_system_prompt(prefs)
    only_significant = getattr(args, "skip_quiet", False)

    drafts, summaries, _failed = await batch_draft(
        client, model, system_prompt, date_dir,
        manifest["issues"], usage,
        only_significant=only_significant,
    )

    issue_keys = [i["key"] for i in manifest["issues"]]
    current_statuses = load_current_statuses(date_dir, issue_keys)

    output_issues = []
    for issue in manifest["issues"]:
        key = issue["key"]
        draft = drafts.get(key, "")
        output_issues.append({
            "key": key,
            "summary": issue.get("summary", ""),
            "assignee": issue.get("assignee"),
            "status": issue.get("status"),
            "draft": draft,
            "color": extract_color(draft) if draft else "",
            "text_summary": summaries.get(key, ""),
            "previous_status": current_statuses.get(key, ""),
        })

    return {
        "manifest": manifest,
        "issues": output_issues,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "calls": usage.calls,
            "model": model,
        },
    }


# ─── Textual TUI ─────────────────────────────────────────────────────────────

class ReviewScreen(Screen):
    """Interactive review: DataTable + new draft pane + previous status pane."""

    CSS = """
    DataTable {
        height: 1fr;
        min-height: 5;
    }
    #draft-panel {
        height: 1fr;
        border: tall $accent;
        padding: 1;
    }
    #current-panel {
        height: 1fr;
        border: tall $surface;
        padding: 1;
        color: $text-muted;
    }
    #request-input {
        display: none;
        height: 3;
        border: tall $warning;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        background: $accent;
        color: $text;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("a", "approve", "Approve", show=True),
        Binding("s", "skip", "Skip", show=True),
        Binding("e", "edit", "Edit in $EDITOR", show=True),
        Binding("i", "interactive_changes", "Interactive changes", show=True),
        Binding("r", "refresh", "Refresh issue & re-draft", show=True),
        Binding("R", "refresh_all", "Refresh all & re-draft", show=True),
        Binding("q", "quit_review", "Quit", show=True),
    ]

    def __init__(
        self, issues, drafts, summaries, current_statuses,
        client, model, usage, jira_url, auth_headers,
        date_dir: Optional[Path] = None,
        prefs: Optional[UserPrefs] = None,
        scripts_dir: Optional[Path] = None,
        args=None,
        system_prompt: str = "",
    ):
        super().__init__()
        self.issues = issues
        self.drafts: dict[str, str] = dict(drafts)
        self.summaries = summaries
        self.current_statuses = current_statuses
        self.client = client
        self.model = model
        self.usage = usage
        self.jira_url = jira_url
        self.auth_headers = auth_headers
        self.date_dir = date_dir
        self.prefs = prefs
        self.scripts_dir = scripts_dir
        self.args = args
        self.system_prompt = system_prompt
        self.results: dict[str, ReviewResult] = {}
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None
        self._col_keys = None

    def _save_drafts(self) -> None:
        if self.date_dir:
            save_drafts(self.date_dir, self.drafts)

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="issue-table")
        yield RichLog(id="draft-panel", highlight=True, markup=True)
        yield RichLog(id="current-panel", highlight=True, markup=True)
        yield Input(placeholder="Describe the changes you want...", id="request-input")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#issue-table", DataTable)
        table.cursor_type = "row"
        self._col_keys = table.add_columns("Key", "Summary", "Color", "Action")
        for issue in self.issues:
            key = issue["key"]
            draft = self.drafts.get(key, "")
            color = extract_color(draft) if draft else "-"
            table.add_row(
                key,
                (issue.get("summary") or "")[:60],
                color,
                "-",
                key=key,
            )
        if self.issues:
            table.move_cursor(row=0)
            self._update_panels()
        self._update_status_bar()

    def on_data_table_row_highlighted(self) -> None:
        self._update_panels()

    def _current_key(self) -> Optional[str]:
        table = self.query_one("#issue-table", DataTable)
        if table.cursor_row is None:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            return row_key.value
        except Exception:
            return None

    def _update_panels(self) -> None:
        key = self._current_key()
        if not key:
            return

        draft = self.drafts.get(key, "(no draft)")
        draft_panel = self.query_one("#draft-panel", RichLog)
        draft_panel.clear()
        draft_panel.write(f"[bold]New draft — {key}[/bold]\n\n{draft}")

        previous = self.current_statuses.get(key, "")
        current_panel = self.query_one("#current-panel", RichLog)
        current_panel.clear()
        if previous:
            current_panel.write(f"[bold]Previous status — {key}[/bold]\n\n{previous}")
        else:
            current_panel.write(f"[bold]Previous status — {key}[/bold]\n\n[dim](none filed)[/dim]")

    def _update_draft_panel(self) -> None:
        self._update_panels()

    def _update_status_bar(self) -> None:
        approved = sum(
            1 for r in self.results.values()
            if r.action in ("approved", "edited", "revised")
        )
        skipped = sum(1 for r in self.results.values() if r.action == "skipped")
        total = len(self.issues)
        self.query_one("#status-bar", Static).update(
            f"  {approved + skipped}/{total} reviewed  |  "
            f"{approved} approved  {skipped} skipped  |  "
            f"a=approve  s=skip  e=edit  i=revise  r=refresh  R=refresh all  q=quit"
        )

    def _mark_action(self, key: str, label: str) -> None:
        table = self.query_one("#issue-table", DataTable)
        table.update_cell(key, self._col_keys[3], label)
        self._update_status_bar()

    def _advance(self) -> None:
        table = self.query_one("#issue-table", DataTable)
        if table.cursor_row is not None and table.cursor_row < table.row_count - 1:
            table.move_cursor(row=table.cursor_row + 1)
        self._update_draft_panel()

    def action_approve(self) -> None:
        key = self._current_key()
        if not key:
            return
        draft = self.drafts.get(key, "")
        color = extract_color(draft)
        self.results[key] = ReviewResult(key=key, action="approved", color=color, text=draft)
        self._mark_action(key, "Approved")
        self.run_worker(self._do_jira_update(key, draft), exclusive=False)
        self._advance()

    def action_skip(self) -> None:
        key = self._current_key()
        if not key:
            return
        draft = self.drafts.get(key, "")
        color = extract_color(draft)
        self.results[key] = ReviewResult(key=key, action="skipped", color=color, text=draft)
        self._mark_action(key, "Skipped")
        self._advance()

    def action_edit(self) -> None:
        key = self._current_key()
        if not key:
            return
        draft = self.drafts.get(key, "")
        with self.app.suspend():
            edited = edit_in_editor(draft)
        if edited:
            self.drafts[key] = edited
            self._save_drafts()
            table = self.query_one("#issue-table", DataTable)
            table.update_cell(key, self._col_keys[2], extract_color(edited))
            self._update_draft_panel()

    def action_interactive_changes(self) -> None:
        inp = self.query_one("#request-input", Input)
        inp.display = True
        inp.focus()

    def action_refresh(self) -> None:
        key = self._current_key()
        if not key or not self.scripts_dir or not self.args:
            self.app.notify("Refresh not available", severity="error")
            return
        self.run_worker(self._do_refresh(key), exclusive=False)

    async def _do_refresh(self, key: str) -> None:
        draft_panel = self.query_one("#draft-panel", RichLog)
        draft_panel.loading = True
        table = self.query_one("#issue-table", DataTable)
        table.update_cell(key, self._col_keys[3], "Refreshing…")
        try:
            _, manifest = await run_gather(
                self.scripts_dir, self.args,
                self.args.output_dir if hasattr(self.args, "output_dir") else ".work/weekly-status",
                refresh=True, issue_keys=[key],
            )

            issue_file = self.date_dir / "issues" / f"{key}.json"
            if not issue_file.exists():
                self.app.notify(f"No data file for {key} after refresh", severity="error")
                return

            issue_data = json.loads(issue_file.read_text())
            pruned = _prune_issue(issue_data)
            summary = json.dumps(pruned, indent=2)
            self.summaries[key] = summary

            table.update_cell(key, self._col_keys[3], "Drafting…")
            draft = await draft_ryg(
                self.client, self.model, self.system_prompt, summary, self.usage,
            )
            self.drafts[key] = draft
            self._save_drafts()
            table.update_cell(key, self._col_keys[2], extract_color(draft))
            table.update_cell(key, self._col_keys[3], "-")
            self.results.pop(key, None)
            self._update_draft_panel()
            self.app.notify(f"{key} refreshed")
        except Exception as e:
            self.app.notify(str(e), title="Refresh failed", severity="error")
        finally:
            draft_panel.loading = False

    def action_refresh_all(self) -> None:
        if not self.scripts_dir or not self.args:
            self.app.notify("Refresh not available", severity="error")
            return
        self.run_worker(self._do_refresh_all(), exclusive=False)

    async def _do_refresh_all(self) -> None:
        draft_panel = self.query_one("#draft-panel", RichLog)
        draft_panel.loading = True
        table = self.query_one("#issue-table", DataTable)
        for issue in self.issues:
            table.update_cell(issue["key"], self._col_keys[3], "Refreshing…")
        try:
            output_dir = getattr(self.args, "output_dir", ".work/weekly-status")
            date_dir, manifest = await run_gather(
                self.scripts_dir, self.args, output_dir,
                refresh=True,
            )
            self.date_dir = date_dir
            for issue in self.issues:
                table.update_cell(issue["key"], self._col_keys[3], "Drafting…")

            if self.date_dir:
                drafts_path = _drafts_path(self.date_dir)
                if drafts_path.exists():
                    drafts_path.unlink()

            drafts, summaries, failed = await batch_draft(
                self.client, self.model, self.system_prompt, date_dir,
                manifest["issues"], self.usage,
            )
            self.drafts.update(drafts)
            if failed:
                self.app.notify(
                    f"Failed to draft: {', '.join(failed)}", title="Some drafts failed", severity="warning",
                )
            self.summaries.update(summaries)
            self._save_drafts()
            for issue in self.issues:
                key = issue["key"]
                draft = self.drafts.get(key, "")
                table.update_cell(key, self._col_keys[2], extract_color(draft) if draft else "-")
                table.update_cell(key, self._col_keys[3], "-")
                self.results.pop(key, None)
            self._update_panels()
            self.app.notify(f"All {len(manifest['issues'])} issues refreshed")
        except Exception as e:
            self.app.notify(str(e), title="Refresh all failed", severity="error")
        finally:
            draft_panel.loading = False

    def on_input_submitted(self, event: Input.Submitted) -> None:
        instruction = event.value.strip()
        inp = self.query_one("#request-input", Input)
        inp.display = False
        inp.clear()
        if not instruction:
            return
        key = self._current_key()
        if not key:
            return

        draft = self.drafts.get(key, "")
        summary = self.summaries.get(key, "")

        agent = _resolve_coding_agent(self.prefs or UserPrefs())
        if agent:
            try:
                with self.app.suspend():
                    revised = _run_agent_interactive(
                        agent, self.model, draft, summary, instruction,
                        issue_key=key, date_dir=self.date_dir,
                    )
                self.drafts[key] = revised
                self._save_drafts()
                table = self.query_one("#issue-table", DataTable)
                table.update_cell(key, self._col_keys[2], extract_color(revised))
                self._update_draft_panel()
            except Exception as e:
                self.app.notify(str(e), title="Revision failed", severity="error")
        else:
            self.run_worker(
                self._do_request_changes(key, draft, summary, instruction),
                exclusive=False,
            )

    async def _do_request_changes(self, key, draft, summary, instruction):
        try:
            revised = await request_changes_api(
                self.client, self.model, draft, summary, instruction, self.usage,
            )
            self.drafts[key] = revised
            self._save_drafts()
            table = self.query_one("#issue-table", DataTable)
            table.update_cell(key, self._col_keys[2], extract_color(revised))
            self._update_draft_panel()
        except Exception as e:
            self.app.notify(str(e), title="Revision failed", severity="error")

    async def _do_jira_update(self, key: str, draft: str) -> None:
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            self._aiohttp_session = aiohttp.ClientSession()
        result = await update_jira_status(
            self._aiohttp_session, self.jira_url, self.auth_headers, key, draft
        )
        table = self.query_one("#issue-table", DataTable)
        if result.ok:
            table.update_cell(key, self._col_keys[3], "✓ Submitted")
        else:
            table.update_cell(key, self._col_keys[3], "✗ FAILED")
            self.app.notify(f"{key}: {result.error}", title="Jira update failed", severity="error")

    def action_quit_review(self) -> None:
        if self._aiohttp_session and not self._aiohttp_session.closed:
            self.run_worker(self._aiohttp_session.close(), exclusive=False)
        self.dismiss(list(self.results.values()))


class StatusUpdateApp(App):
    """Full-pipeline TUI: gather → draft → interactive review → summary."""

    CSS = """
    #phase-label {
        height: 1;
        padding: 0 1;
        background: $accent;
        color: $text;
    }
    #progress {
        height: 1;
        margin: 0 1;
    }
    #log {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, args, prefs: UserPrefs, scripts_dir: Path):
        super().__init__()
        self.args = args
        self.prefs = prefs
        self.scripts_dir = scripts_dir

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Initializing...", id="phase-label")
        yield ProgressBar(id="progress", total=100)
        yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._run_pipeline(), exclusive=True)

    def _set_phase(self, phase: str, total: int = 100) -> None:
        self.query_one("#phase-label", Static).update(f"[bold]{phase}[/bold]")
        pb = self.query_one("#progress", ProgressBar)
        pb.total = total
        pb.progress = 0

    def _log(self, message: str) -> None:
        self.query_one(RichLog).write(message)

    def _make_callback(self):
        async def callback(current: int, total: int, message: str) -> None:
            pb = self.query_one("#progress", ProgressBar)
            pb.total = total
            pb.progress = current
            self.query_one(RichLog).write(message)
        return callback

    async def _run_pipeline(self) -> None:
        args = self.args
        prefs = self.prefs

        try:
            self._set_phase("Gathering data...")
            self._log("Starting data gather...")

            async def _gather_log(line: str) -> None:
                self._log(line)

            refresh = getattr(args, "refresh", False)
            date_dir, manifest = await run_gather(
                self.scripts_dir, args, args.output_dir,
                line_callback=_gather_log,
                refresh=refresh,
            )
            issue_count = len(manifest["issues"])
            self._log(f"[green]✓[/green] Gathered {issue_count} issues")

            self._set_phase("Drafting status updates...", total=issue_count)
            self._log("Starting batch LLM drafting...")
            client = create_client()
            model = args.model or prefs.model
            usage = TokenUsage()
            system_prompt = build_system_prompt(prefs)
            only_significant = getattr(args, "skip_quiet", False)

            drafts, summaries, failed = await batch_draft(
                client, model, system_prompt, date_dir,
                manifest["issues"], usage,
                callback=self._make_callback(),
                only_significant=only_significant,
            )
            self._log(
                f"[green]✓[/green] Drafted {len(drafts)} issues ({usage.calls} API calls)"
            )
            if failed:
                self._log(
                    f"[yellow]⚠[/yellow] Failed to draft {len(failed)} issues: {', '.join(failed)}"
                )

            current_statuses = load_current_statuses(date_dir, [i["key"] for i in manifest["issues"]])

            self._set_phase("Review")
            try:
                auth_headers = get_jira_auth()
            except ValueError as e:
                self._log(f"[red]Error: {e}[/red]")
                return

            results = await self.push_screen_wait(
                ReviewScreen(
                    manifest["issues"], drafts, summaries, current_statuses,
                    client, model, usage, prefs.jira_url, auth_headers,
                    date_dir=date_dir,
                    prefs=prefs,
                    scripts_dir=self.scripts_dir,
                    args=self.args,
                    system_prompt=system_prompt,
                )
            ) or []

            self._set_phase("Done")
            approved = [r for r in results if r.action in ("approved", "edited", "revised")]
            skipped = [r for r in results if r.action == "skipped"]
            green = sum(1 for r in approved if r.color == "Green")
            yellow = sum(1 for r in approved if r.color == "Yellow")
            red = sum(1 for r in approved if r.color == "Red")
            pricing = fetch_model_pricing(model)
            cost_str = f"${usage.cost(pricing):.4f}" if pricing else "unknown"

            self._log(f"\n[bold]Summary[/bold]")
            self._log(
                f"  Updated: {len(approved)} "
                f"(Green: {green}, Yellow: {yellow}, Red: {red})"
            )
            self._log(f"  Skipped: {len(skipped)}")
            self._log(
                f"  LLM: {usage.calls} calls, "
                f"{usage.input_tokens + usage.output_tokens} tokens, "
                f"{cost_str} ({model})"
            )
            if approved:
                self._log("\n[bold]Updated issues:[/bold]")
                tags = {"Green": "green", "Yellow": "yellow", "Red": "red"}
                for r in approved:
                    tag = tags.get(r.color, "white")
                    self._log(f"  [{tag}]{r.color}[/{tag}] {r.key}")

            self._log("\n[dim]Press q to exit.[/dim]")

        except Exception as e:
            import traceback
            self._log(f"[red]Pipeline failed: {e}[/red]")
            self._log(traceback.format_exc())
            self._log("\n[dim]Press q to exit.[/dim]")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jira weekly status update tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jira-status-update --project OCPSTRAT --component "Hosted Control Planes"
  jira-status-update --project OCPSTRAT --profile ocpstrat-hcp
  jira-status-update --project OCPSTRAT --headless --json-output /tmp/out.json

Environment variables:
  JIRA_USERNAME            Atlassian account email (required)
  JIRA_API_TOKEN           Atlassian API token (required)
  GITHUB_TOKEN             GitHub personal access token (required)
  ANTHROPIC_API_KEY        Direct Anthropic API key
  ANTHROPIC_VERTEX_PROJECT_ID  Use Vertex AI instead of direct API
""",
    )
    parser.add_argument("--project", required=True, help="Jira project key (e.g. OCPSTRAT)")
    parser.add_argument("--component", help="Filter by component")
    parser.add_argument("--label", help="Filter by label")
    parser.add_argument("--assignee", action="append", help="Filter by assignee email (repeatable)")
    parser.add_argument("--exclude", action="append", help="Exclude assignee email (repeatable)")
    parser.add_argument("--days", type=int, default=7, help="Lookback days (default: 7)")
    parser.add_argument("--start-date", dest="start_date", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", dest="end_date", help="End date YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--output-dir", dest="output_dir", default=".work/weekly-status",
        help="Output directory for gathered data (default: .work/weekly-status)",
    )
    parser.add_argument(
        "--skip-quiet", dest="skip_quiet", action="store_true",
        help="Skip issues with no significant activity",
    )
    parser.add_argument("--model", help="Anthropic model (default: from prefs or claude-sonnet-5)")
    parser.add_argument("--profile", help="Named profile from prefs.yaml")
    parser.add_argument("--prefs", type=Path, help="Path to prefs.toml")
    parser.add_argument("--refresh", action="store_true", help="Force re-gather even if cached data exists")
    parser.add_argument("--headless", action="store_true", help="Output JSON only, no interaction")
    parser.add_argument(
        "--json-output", dest="json_output", type=Path,
        help="Write JSON output to file (implies --headless)",
    )
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    prefs = load_prefs(args.prefs)

    if args.profile and args.profile in prefs.profiles:
        profile = prefs.profiles[args.profile]
        for key, value in profile.items():
            if not getattr(args, key, None):
                setattr(args, key, value)

    if args.json_output:
        args.headless = True

    scripts_dir = find_scripts_dir(prefs)

    if args.headless:
        output = await run_headless(args, prefs, scripts_dir)
        json_str = json.dumps(output, indent=2)
        if args.json_output:
            args.json_output.write_text(json_str)
            print(f"Output written to {args.json_output}", file=sys.stderr)
        else:
            print(json_str)
    else:
        app = StatusUpdateApp(args, prefs, scripts_dir)
        await app.run_async()


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()

"""Data gathering subprocess, issue JSON processing, draft persistence, and batch drafting."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import pyadf

from .config import _find_cached_gather, _save_cache_meta
from .llm import TokenUsage, draft_ryg


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


# ─── Draft Persistence ───────────────────────────────────────────────────────

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

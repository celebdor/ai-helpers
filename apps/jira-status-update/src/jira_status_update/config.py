"""Configuration, preferences, path resolution, and gather cache."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tomllib


_SKILL_DIR_REL = Path("plugins/jira/skills/status-analysis")
_AI_HELPERS_REPO = "https://github.com/openshift-eng/ai-helpers"


@dataclass
class UserPrefs:
    model: str = "claude-sonnet-5"
    coding_agent: Optional[str] = None
    writing_rules: list = field(default_factory=list)
    jira_url: str = "https://redhat.atlassian.net"
    skills_dir: Optional[str] = None
    update_mode: str = "replace"    # "replace" or "prepend"
    max_history: int = 0            # 0 = unlimited (only meaningful in prepend mode)
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
        update_mode=jira.get("update_mode", "replace"),
        max_history=jira.get("max_history", 0),
        profiles=data.get("profiles", {}),
    )


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

def _get_data_dir() -> Path:
    """Return the platform-appropriate user data directory for this app."""
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


def _find_repo_skills_dir() -> Optional[Path]:
    """Walk up from this file to find the skills dir in a source tree."""
    d = Path(__file__).resolve().parent
    for _ in range(10):
        d = d.parent
        candidate = d / _SKILL_DIR_REL
        if candidate.exists():
            return candidate
    return None


def find_skills_dir(prefs: Optional[UserPrefs] = None) -> Path:
    """Resolve the status-analysis skills directory using a multi-step search.

    1. [jira] skills_dir in prefs.toml (explicit override)
    2. JIRA_STATUS_UPDATE_SKILLS_DIR env var
    3. Platform data dir — previously auto-fetched copy
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

    dev = _find_repo_skills_dir()
    if dev:
        return dev

    return _fetch_ai_helpers(_get_data_dir())


def find_scripts_dir(prefs: Optional[UserPrefs] = None) -> Path:
    """Return the scripts/ subdirectory of the resolved skills dir."""
    return find_skills_dir(prefs) / "scripts"

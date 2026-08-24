"""Coding agent (Claude Code / OpenCode) integration for interactive draft revision."""

import os
import subprocess as sp
import tempfile
from pathlib import Path
from typing import Optional

from .config import UserPrefs


def _resolve_coding_agent(prefs: UserPrefs) -> Optional[str]:
    """Return the coding agent CLI path from prefs, or None if unset/missing."""
    import shutil
    name = prefs.coding_agent
    if not name:
        return None
    return shutil.which(name)


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

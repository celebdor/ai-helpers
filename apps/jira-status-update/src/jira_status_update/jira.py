"""Jira API interaction: authentication, ADF conversion, and field updates."""

import base64
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

import aiohttp
import pyadf


def _text_to_adf(text: str) -> dict:
    """Convert markdown text into an ADF document for Jira Cloud API v3."""
    return pyadf.Document.from_markdown(text).to_adf()


def _adf_to_text(value) -> str:
    """Convert an ADF dict to markdown text, or pass through strings."""
    if not value:
        return ""
    if isinstance(value, dict):
        try:
            return pyadf.Document(value).to_markdown()
        except Exception:
            return ""
    return str(value)


@dataclass
class JiraUpdateResult:
    ok: bool
    error: Optional[str] = None


@dataclass
class StatusEntry:
    """One dated entry in a prepend-mode status history."""
    date_str: str   # "YYYY-MM-DD"
    body: str       # The status text (without the date header)


_DATE_HEADER_RE = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


def split_entries(text: str) -> list[StatusEntry]:
    """Parse a prepend-mode status field into a list of StatusEntry objects.

    Expected format:
        ## 2026-08-25

        status body...

        ---

        ## 2026-08-18

        older status body...

    Returns entries in document order (newest first).
    """
    if not text or not text.strip():
        return []

    matches = list(_DATE_HEADER_RE.finditer(text))
    if not matches:
        # No date headers — treat entire text as a single undated entry
        return [StatusEntry(date_str="", body=text.strip())]

    entries: list[StatusEntry] = []
    for i, m in enumerate(matches):
        date_str = m.group(1)
        start = m.end()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)
        body = text[start:end].strip()
        # Strip trailing --- separator
        body = re.sub(r"\n*---\s*$", "", body).strip()
        entries.append(StatusEntry(date_str=date_str, body=body))

    return entries


def _assemble_entries(entries: list[StatusEntry]) -> str:
    """Join a list of StatusEntry objects back into formatted text."""
    parts: list[str] = []
    for entry in entries:
        if entry.date_str:
            parts.append(f"## {entry.date_str}\n\n{entry.body}")
        else:
            parts.append(entry.body)
    return "\n\n---\n\n".join(parts)


async def _get_current_field_text(
    session: aiohttp.ClientSession,
    jira_url: str,
    auth_headers: dict,
    issue_key: str,
) -> str:
    """GET the live value of customfield_10814 as markdown text."""
    url = f"{jira_url}/rest/api/3/issue/{issue_key}?fields=customfield_10814"
    async with session.get(url, headers=auth_headers) as resp:
        if resp.status != 200:
            return ""
        data = await resp.json()
        raw = data.get("fields", {}).get("customfield_10814")
        return _adf_to_text(raw)


def _build_prepend_text(
    new_body: str,
    existing_text: str,
    today: Optional[str] = None,
    max_history: int = 0,
) -> str:
    """Build the combined text for prepend mode.

    - Parses existing_text into entries
    - If the top entry is from today, replaces it; otherwise inserts at position 0
    - Trims to max_history entries if max_history > 0
    - Returns the assembled text
    """
    if today is None:
        today = date.today().isoformat()

    existing_entries = split_entries(existing_text)
    new_entry = StatusEntry(date_str=today, body=new_body)

    if existing_entries and existing_entries[0].date_str == today:
        # Same-day dedup: replace top entry
        existing_entries[0] = new_entry
    else:
        existing_entries.insert(0, new_entry)

    if max_history > 0:
        existing_entries = existing_entries[:max_history]

    return _assemble_entries(existing_entries)


async def update_jira_status(
    session: aiohttp.ClientSession,
    jira_url: str,
    auth_headers: dict,
    issue_key: str,
    status_text: str,
    update_mode: str = "replace",
    max_history: int = 0,
) -> JiraUpdateResult:
    """Set customfield_10814 on a Jira issue.

    In "replace" mode (default), overwrites the field entirely.
    In "prepend" mode, top-posts the new status above existing content
    with a date header, building a newest-first chronological log.

    After a successful PUT, reads the field back to verify it actually changed.
    """
    if update_mode == "prepend":
        existing_text = await _get_current_field_text(
            session, jira_url, auth_headers, issue_key
        )
        final_text = _build_prepend_text(
            status_text, existing_text, max_history=max_history
        )
    else:
        final_text = status_text

    url = f"{jira_url}/rest/api/3/issue/{issue_key}"
    payload = {"fields": {"customfield_10814": _text_to_adf(final_text)}}
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

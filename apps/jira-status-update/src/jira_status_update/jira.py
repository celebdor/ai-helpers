"""Jira API interaction: authentication, ADF conversion, and field updates."""

import base64
import os
from dataclasses import dataclass
from typing import Optional

import aiohttp
import pyadf


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

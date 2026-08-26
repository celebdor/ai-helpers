"""Unit tests for jira_status_update — all offline, no API keys required."""

import argparse
import base64
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jira_status_update.app import ReviewResult, edit_in_editor, extract_color
from jira_status_update.config import UserPrefs, find_scripts_dir, find_skills_dir, load_prefs
from jira_status_update.gather import (
    _field_to_text,
    _prune_issue,
    is_significant,
    load_current_statuses,
)
from jira_status_update.jira import (
    StatusEntry,
    _build_prepend_text,
    get_jira_auth,
    split_entries,
)
from jira_status_update.llm import TokenUsage, build_system_prompt


# ─── build_system_prompt ─────────────────────────────────────────────────────

def test_build_system_prompt_includes_skill_docs(tmp_path):
    activity = "# Activity Analysis\n\nGreen means on-track."
    formatting = "# Formatting\n\n* Color Status: Green"
    (tmp_path / "activity-analysis.md").write_text(activity)
    (tmp_path / "formatting.md").write_text(formatting)
    prefs = UserPrefs(writing_rules=["No fractions"])

    with patch("jira_status_update.llm.find_skills_dir", return_value=tmp_path):
        prompt = build_system_prompt(prefs)

    assert activity in prompt
    assert formatting in prompt
    assert "No fractions" in prompt
    # Canonical docs must come before user rules
    assert prompt.index(activity) < prompt.index("No fractions")
    assert "output starts with `* Color Status:`" in prompt


def test_build_system_prompt_no_user_rules(tmp_path):
    (tmp_path / "activity-analysis.md").write_text("analysis")
    (tmp_path / "formatting.md").write_text("formatting")
    prefs = UserPrefs()

    with patch("jira_status_update.llm.find_skills_dir", return_value=tmp_path):
        prompt = build_system_prompt(prefs)

    assert "Additional writing rules" not in prompt


def test_build_system_prompt_missing_files_silently_skipped(tmp_path):
    prefs = UserPrefs()
    with patch("jira_status_update.llm.find_skills_dir", return_value=tmp_path):
        prompt = build_system_prompt(prefs)
    assert "You are a Jira status analyst" in prompt


# ─── load_prefs ──────────────────────────────────────────────────────────────

def test_load_prefs_defaults_when_file_missing(tmp_path):
    prefs = load_prefs(tmp_path / "nonexistent.toml")
    assert prefs.model == "claude-sonnet-5"
    assert prefs.writing_rules == []
    assert prefs.jira_url == "https://redhat.atlassian.net"
    assert prefs.profiles == {}


def test_load_prefs_from_toml(tmp_path):
    content = """
[llm]
model = "claude-haiku-4-5"
writing_rules = ["No fractions", "No percentages"]

[jira]
url = "https://example.atlassian.net"
skills_dir = "/custom/skills"

[profiles.myprofile]
project = "TEST"
"""
    prefs_file = tmp_path / "prefs.toml"
    prefs_file.write_text(content)

    prefs = load_prefs(prefs_file)

    assert prefs.model == "claude-haiku-4-5"
    assert "No fractions" in prefs.writing_rules
    assert prefs.jira_url == "https://example.atlassian.net"
    assert prefs.skills_dir == "/custom/skills"
    assert "myprofile" in prefs.profiles


def test_load_prefs_partial_fills_defaults(tmp_path):
    prefs_file = tmp_path / "prefs.toml"
    prefs_file.write_text("[llm]\nmodel = \"claude-opus-5\"\n")
    prefs = load_prefs(prefs_file)
    assert prefs.model == "claude-opus-5"
    assert prefs.writing_rules == []


# ─── TokenUsage.cost ─────────────────────────────────────────────────────────

def test_token_usage_cost_arithmetic():
    usage = TokenUsage(
        input_tokens=1_000,
        output_tokens=500,
        cache_write_tokens=2_000,
        cache_read_tokens=4_000,
        calls=5,
    )
    pricing = {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30}
    expected = (
        1_000 * 3.0 / 1_000_000
        + 500 * 15.0 / 1_000_000
        + 2_000 * 3.75 / 1_000_000
        + 4_000 * 0.30 / 1_000_000
    )
    assert abs(usage.cost(pricing) - expected) < 1e-10


def test_token_usage_cost_zero():
    pricing = {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30}
    assert TokenUsage().cost(pricing) == 0.0


# ─── get_jira_auth ───────────────────────────────────────────────────────────

def test_get_jira_auth_produces_correct_base64():
    with patch.dict(os.environ, {"JIRA_USERNAME": "user@example.com", "JIRA_API_TOKEN": "tok"}):
        headers = get_jira_auth()
    expected = base64.b64encode(b"user@example.com:tok").decode()
    assert headers["Authorization"] == f"Basic {expected}"
    assert headers["Accept"] == "application/json"
    assert headers["Content-Type"] == "application/json"


def test_get_jira_auth_raises_when_env_missing():
    with patch.dict(os.environ, {"JIRA_USERNAME": "", "JIRA_API_TOKEN": ""}):
        with pytest.raises(ValueError, match="JIRA_USERNAME and JIRA_API_TOKEN"):
            get_jira_auth()


# ─── extract_color ───────────────────────────────────────────────────────────

def test_extract_color_green():
    assert extract_color("* Color Status: Green\nSome text") == "Green"


def test_extract_color_yellow():
    assert extract_color("* Color Status: Yellow\nAt risk") == "Yellow"


def test_extract_color_red():
    assert extract_color("* Color Status: Red\nBlocked") == "Red"


def test_extract_color_defaults_to_green_when_missing():
    assert extract_color("No color line here") == "Green"


def test_extract_color_case_sensitive():
    # Color names are title-case; lowercase doesn't match
    assert extract_color("* Color Status: green") == "Green"  # defaults


# ─── find_scripts_dir / find_skills_dir ──────────────────────────────────────

def test_find_skills_dir_uses_prefs_skills_dir(tmp_path):
    fake_skills = tmp_path / "status-analysis"
    fake_skills.mkdir()
    prefs = UserPrefs(skills_dir=str(fake_skills))
    assert find_skills_dir(prefs) == fake_skills


def test_find_skills_dir_prefs_missing_path_raises(tmp_path):
    prefs = UserPrefs(skills_dir=str(tmp_path / "nonexistent"))
    with pytest.raises(FileNotFoundError, match="skills_dir from prefs not found"):
        find_skills_dir(prefs)


def test_find_skills_dir_env_var(tmp_path):
    fake_skills = tmp_path / "status-analysis"
    fake_skills.mkdir()
    with patch.dict(os.environ, {"JIRA_STATUS_UPDATE_SKILLS_DIR": str(fake_skills)}):
        assert find_skills_dir() == fake_skills


def test_find_skills_dir_env_var_missing_raises(tmp_path):
    with patch.dict(os.environ, {"JIRA_STATUS_UPDATE_SKILLS_DIR": str(tmp_path / "gone")}):
        with pytest.raises(FileNotFoundError, match="JIRA_STATUS_UPDATE_SKILLS_DIR not found"):
            find_skills_dir()


def test_find_skills_dir_dev_fallback():
    # Running inside the source tree — repo-relative heuristic should resolve.
    skills_dir = find_skills_dir()
    assert skills_dir.name == "status-analysis"
    assert skills_dir.parent.name == "skills"


def test_find_scripts_dir_returns_scripts_subdir(tmp_path):
    fake_skills = tmp_path / "status-analysis"
    (fake_skills / "scripts").mkdir(parents=True)
    prefs = UserPrefs(skills_dir=str(fake_skills))
    assert find_scripts_dir(prefs).name == "scripts"
    assert find_scripts_dir(prefs).parent == fake_skills


# ─── edit_in_editor ──────────────────────────────────────────────────────────

def test_edit_in_editor_returns_edited_text():
    expected = "This is the edited draft."

    def fake_run(*args, **kwargs):
        tmpfile = args[0][1]
        with open(tmpfile, "w") as f:
            f.write(expected + "\n")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        with patch.dict(os.environ, {"EDITOR": "vi"}):
            result = edit_in_editor("original draft")

    assert result == expected


def test_edit_in_editor_strips_comment_lines():
    def fake_run(*args, **kwargs):
        tmpfile = args[0][1]
        with open(tmpfile, "w") as f:
            f.write("Actual content\n# This comment should be stripped\n")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        with patch.dict(os.environ, {"EDITOR": "vi"}):
            result = edit_in_editor("original")

    assert result == "Actual content"
    assert "comment" not in result


def test_edit_in_editor_returns_none_on_empty_file():
    def fake_run(*args, **kwargs):
        tmpfile = args[0][1]
        with open(tmpfile, "w") as f:
            f.write("")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        with patch.dict(os.environ, {"EDITOR": "vi"}):
            result = edit_in_editor("original")

    assert result is None


# ─── _prune_issue / is_significant ───────────────────────────────────────────

def _make_issue_data(**kwargs):
    """Minimal per-issue JSON dict for testing."""
    base = {
        "issue": {
            "key": "ISSUE-1",
            "summary": "Test issue",
            "status": "In Progress",
            "assignee": {"name": "Alice", "email": "alice@example.com"},
            "current_status_summary": None,
            "last_status_summary_update": None,
        },
        "descendants": {"total": 0, "completion_pct": 0.0, "by_status": {}, "updated_in_range": []},
        "changelog_in_range": [],
        "comments_in_range": [],
        "prs": [],
    }
    base.update(kwargs)
    return base


def test_prune_issue_includes_key_fields():
    data = _make_issue_data()
    pruned = _prune_issue(data)
    assert pruned["key"] == "ISSUE-1"
    assert pruned["summary"] == "Test issue"
    assert pruned["status"] == "In Progress"
    assert pruned["assignee"] == "Alice"
    assert "descendants" in pruned
    assert "changelog" in pruned
    assert "comments" in pruned
    assert "prs" in pruned


def test_prune_issue_converts_adf_current_status():
    adf = {"type": "doc", "version": 1, "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": "Green - on track"}]}
    ]}
    data = _make_issue_data()
    data["issue"]["current_status_summary"] = adf
    pruned = _prune_issue(data)
    assert "Green - on track" in pruned["current_status_summary"]
    assert isinstance(pruned["current_status_summary"], str)


def test_prune_issue_strips_bot_comments():
    data = _make_issue_data()
    data["comments_in_range"] = [
        {"date": "2026-08-01T00:00:00", "author_name": "bot", "body": "automated", "is_bot": True},
        {"date": "2026-08-01T00:00:00", "author_name": "human", "body": "real comment", "is_bot": False},
    ]
    pruned = _prune_issue(data)
    assert len(pruned["comments"]) == 1
    assert pruned["comments"][0]["author_name"] == "human"


def test_prune_issue_pr_drops_raw_arrays():
    data = _make_issue_data()
    data["prs"] = [{
        "number": 42,
        "title": "Fix bug",
        "state": "MERGED",
        "is_draft": False,
        "merged_at": "2026-08-01T12:00:00Z",
        "dates": {"merged_at": "2026-08-01T12:00:00Z"},
        "activity_summary": {"commits_in_range": 3, "reviews_in_range": 1, "review_comments_in_range": 2},
        "commits_in_range": [{"sha": "abc123", "author_name": "Alice"}],
        "reviews_in_range": [{"author_name": "Bob"}],
    }]
    pruned = _prune_issue(data)
    pr = pruned["prs"][0]
    assert pr["number"] == 42
    assert pr["commits_in_range"] == 3
    assert pr["reviews_in_range"] == 1
    assert isinstance(pr["commits_in_range"], int)
    assert "sha" not in str(pruned)


def test_is_significant_active_pr():
    data = _make_issue_data()
    data["prs"] = [{"state": "OPEN", "activity_summary": {"commits_in_range": 2, "reviews_in_range": 0}}]
    assert is_significant(data) is True


def test_is_significant_no_activity():
    data = _make_issue_data()
    assert is_significant(data) is False


def test_field_to_text_plain_string():
    assert _field_to_text("hello") == "hello"


def test_field_to_text_adf_dict():
    adf = {"type": "doc", "version": 1, "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": "status text"}]}
    ]}
    result = _field_to_text(adf)
    assert "status text" in result


# ─── load_current_statuses ───────────────────────────────────────────────────

def test_load_current_statuses_reads_from_json(tmp_path):
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    (issues_dir / "ISSUE-1.json").write_text(json.dumps({
        "issue": {"key": "ISSUE-1", "current_status_summary": "Green - on track"}
    }))
    (issues_dir / "ISSUE-2.json").write_text(json.dumps({
        "issue": {"key": "ISSUE-2", "current_status_summary": ""}
    }))

    statuses = load_current_statuses(tmp_path, ["ISSUE-1", "ISSUE-2", "ISSUE-3"])

    assert statuses["ISSUE-1"] == "Green - on track"
    assert statuses["ISSUE-2"] == ""
    assert statuses["ISSUE-3"] == ""  # missing file → empty string


def test_load_current_statuses_handles_missing_field(tmp_path):
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    (issues_dir / "ISSUE-1.json").write_text(json.dumps({
        "issue": {"key": "ISSUE-1"}  # no current_status_summary key
    }))

    statuses = load_current_statuses(tmp_path, ["ISSUE-1"])
    assert statuses["ISSUE-1"] == ""


def test_load_current_statuses_converts_adf_to_markdown(tmp_path):
    issues_dir = tmp_path / "issues"
    issues_dir.mkdir()
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": "Green - on track"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    (issues_dir / "ISSUE-1.json").write_text(json.dumps({
        "issue": {"key": "ISSUE-1", "current_status_summary": adf}
    }))

    statuses = load_current_statuses(tmp_path, ["ISSUE-1"])
    assert "Green - on track" in statuses["ISSUE-1"]
    assert isinstance(statuses["ISSUE-1"], str)


# ─── Profile CLI override ────────────────────────────────────────────────────

def test_profile_values_fill_missing_cli_args():
    prefs = UserPrefs(
        profiles={"my-profile": {"component": "Profile Component", "label": "profile-label"}}
    )
    args = argparse.Namespace(
        project="TEST",
        component="CLI Component",  # explicitly set — must NOT be overridden
        label=None,                  # not set — profile should fill this in
        profile="my-profile",
    )

    if args.profile and args.profile in prefs.profiles:
        for key, value in prefs.profiles[args.profile].items():
            if not getattr(args, key, None):
                setattr(args, key, value)

    assert args.component == "CLI Component"   # CLI wins
    assert args.label == "profile-label"       # profile fills missing


def test_profile_does_not_override_explicitly_set_args():
    prefs = UserPrefs(
        profiles={"p": {"component": "Profile", "label": "plabel", "model": "claude-haiku-4-5"}}
    )
    args = argparse.Namespace(
        project="TEST",
        component="My Component",
        label="my-label",
        model=None,
        profile="p",
    )

    if args.profile and args.profile in prefs.profiles:
        for key, value in prefs.profiles[args.profile].items():
            if not getattr(args, key, None):
                setattr(args, key, value)

    assert args.component == "My Component"
    assert args.label == "my-label"
    assert args.model == "claude-haiku-4-5"  # profile fills model (was None)


# ─── split_entries ──────────────────────────────────────────────────────────

def test_split_entries_empty_string():
    assert split_entries("") == []


def test_split_entries_whitespace_only():
    assert split_entries("   \n  ") == []


def test_split_entries_no_date_headers():
    """Text without ## headers is treated as a single undated entry."""
    text = "* Color Status: Green\n\nAll good."
    entries = split_entries(text)
    assert len(entries) == 1
    assert entries[0].date_str == ""
    assert "Color Status: Green" in entries[0].body


def test_split_entries_single_entry():
    text = "## 2026-08-25\n\n* Color Status: Green\n\nAll good."
    entries = split_entries(text)
    assert len(entries) == 1
    assert entries[0].date_str == "2026-08-25"
    assert "Color Status: Green" in entries[0].body


def test_split_entries_multiple_entries():
    text = (
        "## 2026-08-25\n\nNew status\n\n---\n\n"
        "## 2026-08-18\n\nOlder status\n\n---\n\n"
        "## 2026-08-11\n\nOldest status"
    )
    entries = split_entries(text)
    assert len(entries) == 3
    assert entries[0].date_str == "2026-08-25"
    assert "New status" in entries[0].body
    assert entries[1].date_str == "2026-08-18"
    assert "Older status" in entries[1].body
    assert entries[2].date_str == "2026-08-11"
    assert "Oldest status" in entries[2].body


def test_split_entries_strips_trailing_separator():
    text = "## 2026-08-25\n\nSome status\n\n---"
    entries = split_entries(text)
    assert len(entries) == 1
    assert entries[0].body == "Some status"
    assert "---" not in entries[0].body


def test_split_entries_preserves_body_content():
    """Multiline body with bullet points is preserved."""
    body = "* Color Status: Yellow\n\n* Risk: deadline slip\n* Mitigation: extra sprint"
    text = f"## 2026-08-25\n\n{body}"
    entries = split_entries(text)
    assert entries[0].body == body


# ─── _build_prepend_text ────────────────────────────────────────────────────

def test_build_prepend_text_empty_existing():
    result = _build_prepend_text("New status", "", today="2026-08-25")
    assert result == "## 2026-08-25\n\nNew status"


def test_build_prepend_text_prepends_before_existing():
    existing = "## 2026-08-18\n\nOld status"
    result = _build_prepend_text("New status", existing, today="2026-08-25")
    assert result.startswith("## 2026-08-25\n\nNew status")
    assert "---" in result
    assert "## 2026-08-18\n\nOld status" in result
    # New entry must come before old
    assert result.index("2026-08-25") < result.index("2026-08-18")


def test_build_prepend_text_same_day_dedup():
    """If top entry is from today, replace it instead of adding a new one."""
    existing = "## 2026-08-25\n\nMorning status\n\n---\n\n## 2026-08-18\n\nLast week"
    result = _build_prepend_text("Afternoon status", existing, today="2026-08-25")
    entries = split_entries(result)
    assert len(entries) == 2
    assert entries[0].date_str == "2026-08-25"
    assert "Afternoon status" in entries[0].body
    assert "Morning status" not in result
    assert entries[1].date_str == "2026-08-18"


def test_build_prepend_text_max_history_trims():
    existing = (
        "## 2026-08-18\n\nWeek 3\n\n---\n\n"
        "## 2026-08-11\n\nWeek 2\n\n---\n\n"
        "## 2026-08-04\n\nWeek 1"
    )
    result = _build_prepend_text("Week 4", existing, today="2026-08-25", max_history=3)
    entries = split_entries(result)
    assert len(entries) == 3
    assert entries[0].date_str == "2026-08-25"
    assert entries[1].date_str == "2026-08-18"
    assert entries[2].date_str == "2026-08-11"
    # Week 1 (2026-08-04) should be trimmed
    assert "2026-08-04" not in result


def test_build_prepend_text_max_history_zero_unlimited():
    # Use valid dates spanning several months
    dates = [
        "2026-08-25", "2026-08-18", "2026-08-11", "2026-08-04",
        "2026-07-28", "2026-07-21", "2026-07-14", "2026-07-07",
        "2026-06-29", "2026-06-22",
    ]
    existing = "\n\n---\n\n".join(
        f"## {d}\n\nWeek {i}" for i, d in enumerate(dates)
    )
    result = _build_prepend_text("Latest", existing, today="2026-09-01", max_history=0)
    entries = split_entries(result)
    assert len(entries) == 11  # 10 existing + 1 new


def test_build_prepend_text_same_day_dedup_with_max_history():
    """Same-day dedup + max_history should not double-count the replaced entry."""
    existing = (
        "## 2026-08-25\n\nEarlier today\n\n---\n\n"
        "## 2026-08-18\n\nLast week"
    )
    result = _build_prepend_text("Updated today", existing, today="2026-08-25", max_history=2)
    entries = split_entries(result)
    assert len(entries) == 2
    assert entries[0].date_str == "2026-08-25"
    assert "Updated today" in entries[0].body
    assert entries[1].date_str == "2026-08-18"


def test_build_prepend_text_undated_existing():
    """Existing content without date headers is preserved after the new entry."""
    existing = "* Color Status: Green\n\nLegacy status"
    result = _build_prepend_text("New status", existing, today="2026-08-25")
    # New dated entry appears first, then separator, then legacy content
    assert result.startswith("## 2026-08-25\n\nNew status")
    assert "---" in result
    assert "Legacy status" in result
    # New content comes before legacy content
    assert result.index("New status") < result.index("Legacy status")


# ─── config: update_mode and max_history ────────────────────────────────────

def test_user_prefs_defaults():
    prefs = UserPrefs()
    assert prefs.update_mode == "replace"
    assert prefs.max_history == 0


def test_load_prefs_reads_update_mode(tmp_path):
    content = """
[jira]
update_mode = "prepend"
max_history = 10
"""
    prefs_file = tmp_path / "prefs.toml"
    prefs_file.write_text(content)
    prefs = load_prefs(prefs_file)
    assert prefs.update_mode == "prepend"
    assert prefs.max_history == 10


def test_load_prefs_update_mode_defaults_when_missing(tmp_path):
    content = "[jira]\nurl = \"https://example.atlassian.net\"\n"
    prefs_file = tmp_path / "prefs.toml"
    prefs_file.write_text(content)
    prefs = load_prefs(prefs_file)
    assert prefs.update_mode == "replace"
    assert prefs.max_history == 0


def test_profile_can_set_update_mode():
    """Profiles can override update_mode, resolved via the standard profile loop."""
    prefs = UserPrefs(
        profiles={"team-a": {"update_mode": "prepend", "max_history": 5}}
    )
    args = argparse.Namespace(
        project="TEST", update_mode=None, profile="team-a",
    )

    if args.profile and args.profile in prefs.profiles:
        for key, value in prefs.profiles[args.profile].items():
            if not getattr(args, key, None):
                setattr(args, key, value)

    assert args.update_mode == "prepend"


def test_cli_flag_overrides_profile_update_mode():
    """CLI --update-mode flag takes priority over profile setting."""
    prefs = UserPrefs(
        profiles={"team-a": {"update_mode": "prepend"}}
    )
    args = argparse.Namespace(
        project="TEST", update_mode="replace", profile="team-a",
    )

    if args.profile and args.profile in prefs.profiles:
        for key, value in prefs.profiles[args.profile].items():
            if not getattr(args, key, None):
                setattr(args, key, value)

    assert args.update_mode == "replace"  # CLI wins

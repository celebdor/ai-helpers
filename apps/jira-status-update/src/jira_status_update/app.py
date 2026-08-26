"""Textual TUI: ReviewScreen for interactive review and StatusUpdateApp for the full pipeline."""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import aiohttp

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, ProgressBar, RichLog, Static

from .agent import _resolve_coding_agent, _run_agent_interactive
from .config import UserPrefs
from .gather import (
    _drafts_path,
    _prune_issue,
    batch_draft,
    load_current_statuses,
    run_gather,
    save_drafts,
)
from .jira import get_jira_auth, split_entries, update_jira_status
from .llm import (
    TokenUsage,
    build_system_prompt,
    create_client,
    draft_ryg,
    fetch_model_pricing,
    request_changes_api,
)


def extract_color(draft: str) -> str:
    """Extract R/Y/G color from a draft status update. Defaults to Green."""
    for line in draft.split("\n"):
        if "Color Status:" in line:
            for color in ("Red", "Yellow", "Green"):
                if color in line:
                    return color
    return "Green"


@dataclass
class ReviewResult:
    key: str
    action: str   # "approved", "skipped", "edited", "revised"
    color: str    # "Green", "Yellow", "Red"
    text: str


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
        print(f"Editor failed: {e}", file=__import__("sys").stderr)
        os.unlink(tmppath)
        return None

    with open(tmppath) as f:
        lines = [line for line in f.readlines() if not line.startswith("#")]
    os.unlink(tmppath)
    result = "".join(lines).strip()
    return result if result else None


class DedupConfirmScreen(ModalScreen[bool]):
    """Modal confirmation dialog for same-day dedup in prepend mode."""

    CSS = """
    DedupConfirmScreen {
        align: center middle;
    }
    #dedup-dialog {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #dedup-dialog Label {
        width: 100%;
        margin-bottom: 1;
    }
    #dedup-buttons {
        width: 100%;
        height: auto;
        align: center middle;
    }
    #dedup-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, issue_key: str, existing_date: str) -> None:
        super().__init__()
        self.issue_key = issue_key
        self.existing_date = existing_date

    def compose(self) -> ComposeResult:
        with Static(id="dedup-dialog"):
            yield Label(
                f"A status entry dated {self.existing_date} already exists "
                f"for {self.issue_key}. Replace today's entry or cancel?"
            )
            with Horizontal(id="dedup-buttons"):
                yield Button("Replace today's entry", variant="warning", id="btn-replace")
                yield Button("Cancel", variant="default", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-replace")


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
        update_mode: str = "replace",
        max_history: int = 0,
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
        self.update_mode = update_mode
        self.max_history = max_history
        self.results: dict[str, ReviewResult] = {}
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None
        self._col_keys = None

    @property
    def _is_prepend(self) -> bool:
        return self.update_mode == "prepend"

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

    def _build_prepend_preview(self, key: str, draft: str) -> str:
        """Build a combined preview for prepend mode: new entry + existing content."""
        today = date.today().isoformat()
        existing = self.current_statuses.get(key, "")
        entries = split_entries(existing)

        # Show what the field will look like after approve
        preview_parts = [f"## {today}\n\n{draft}"]
        for entry in entries:
            if entry.date_str == today:
                continue  # Will be replaced by the new entry
            header = f"## {entry.date_str}" if entry.date_str else ""
            if header:
                preview_parts.append(f"{header}\n\n{entry.body}")
            else:
                preview_parts.append(entry.body)
        return "\n\n---\n\n".join(preview_parts)

    def _update_panels(self) -> None:
        key = self._current_key()
        if not key:
            return

        draft = self.drafts.get(key, "(no draft)")

        # Draft panel — left pane
        draft_panel = self.query_one("#draft-panel", RichLog)
        draft_panel.clear()
        mode_tag = " [prepend]" if self._is_prepend else ""
        draft_panel.write(f"[bold]New draft — {key}{mode_tag}[/bold]\n\n{draft}")

        # Right pane — current status or result preview
        current_panel = self.query_one("#current-panel", RichLog)
        current_panel.clear()

        if self._is_prepend:
            preview = self._build_prepend_preview(key, draft)
            current_panel.write(f"[bold]Result Preview — {key}[/bold]\n\n{preview}")
        else:
            previous = self.current_statuses.get(key, "")
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
        approve_label = "a=approve (prepend)" if self._is_prepend else "a=approve"
        self.query_one("#status-bar", Static).update(
            f"  {approved + skipped}/{total} reviewed  |  "
            f"{approved} approved  {skipped} skipped  |  "
            f"{approve_label}  s=skip  e=edit  i=revise  r=refresh  R=refresh all  q=quit"
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

    def _do_approve(self, key: str) -> None:
        """Finalize approval: record result, fire Jira update, advance."""
        draft = self.drafts.get(key, "")
        color = extract_color(draft)
        self.results[key] = ReviewResult(key=key, action="approved", color=color, text=draft)
        self._mark_action(key, "Approved")
        self.run_worker(self._do_jira_update(key, draft), exclusive=False)
        self._advance()

    def action_approve(self) -> None:
        key = self._current_key()
        if not key:
            return

        if self._is_prepend:
            # Check for same-day dedup
            existing = self.current_statuses.get(key, "")
            entries = split_entries(existing)
            today = date.today().isoformat()
            if entries and entries[0].date_str == today:
                # Show warning and confirmation dialog
                self._mark_action(key, f"⚠ Already has status from today")
                self.app.push_screen(
                    DedupConfirmScreen(key, today),
                    callback=lambda confirmed: self._on_dedup_confirmed(key, confirmed),
                )
                return

        self._do_approve(key)

    def _on_dedup_confirmed(self, key: str, confirmed: bool) -> None:
        """Handle result from dedup confirmation dialog."""
        if confirmed:
            self._do_approve(key)
        else:
            # Clear the warning and stay on the same issue
            self._mark_action(key, "-")

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
            self._aiohttp_session, self.jira_url, self.auth_headers, key, draft,
            update_mode=self.update_mode,
            max_history=self.max_history,
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
    """Full-pipeline TUI: gather -> draft -> interactive review -> summary."""

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

            # Resolve update_mode and max_history
            update_mode = getattr(args, "update_mode", None) or prefs.update_mode
            max_history = getattr(args, "max_history", None) or prefs.max_history

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
                    update_mode=update_mode,
                    max_history=max_history,
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
            if update_mode == "prepend":
                self._log(f"  Mode: prepend (max_history={max_history or 'unlimited'})")
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

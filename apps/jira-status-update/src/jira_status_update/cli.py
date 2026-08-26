"""CLI entry point: argument parsing and main."""

import argparse
import asyncio
from pathlib import Path

from .app import StatusUpdateApp
from .config import find_scripts_dir, load_prefs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jira weekly status update tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jira-status-update --project OCPSTRAT --component "Hosted Control Planes"
  jira-status-update --project OCPSTRAT --profile ocpstrat-hcp

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
    parser.add_argument("--profile", help="Named profile from prefs.toml")
    parser.add_argument("--prefs", type=Path, help="Path to prefs.toml")
    parser.add_argument("--refresh", action="store_true", help="Force re-gather even if cached data exists")
    parser.add_argument(
        "--update-mode", dest="update_mode", choices=["replace", "prepend"],
        default=None,
        help="How to write status updates: 'replace' overwrites the field, "
             "'prepend' top-posts above existing content (default: from prefs or replace)",
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

    # Resolve update_mode: CLI flag → profile → global [jira] → default
    if not args.update_mode:
        args.update_mode = prefs.update_mode

    scripts_dir = find_scripts_dir(prefs)
    app = StatusUpdateApp(args, prefs, scripts_dir)
    await app.run_async()


def cli() -> None:
    asyncio.run(main())

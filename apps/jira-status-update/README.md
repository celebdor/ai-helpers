# jira-status-update

A TUI app that gathers Jira issue data (PRs, changelogs, comments, descendants), drafts weekly R/Y/G status updates using an LLM, and lets you review, edit, and push them to Jira — all from the terminal.

## Features

- **Automated data gathering** — pulls issue activity from Jira and GitHub (PRs, reviews, commits) for a configurable date range
- **LLM-drafted updates** — produces Red/Yellow/Green status summaries using your project's analysis and formatting rules
- **Interactive TUI review** — side-by-side view of the new draft and previous status for each issue
- **In-app editing** — edit drafts in `$EDITOR`, request AI revisions via the API, or hand off to a coding agent (Claude Code, OpenCode) for tool-assisted changes
- **Single-issue refresh** — re-gather and re-draft one issue without restarting the pipeline
- **Same-day caching** — gathered data and drafts are cached per UTC day; subsequent runs skip re-gathering
- **Profiles** — save named parameter sets (project, component, label, assignees) in `prefs.toml`
- **Cost tracking** — shows token usage and estimated cost at the end of each session

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A Jira Cloud instance with API access
- An Anthropic API key or Google Cloud project with Vertex AI access
- A GitHub personal access token (for PR data)

## Installation

```bash
uv tool install --from ./apps/jira-status-update jira-status-update
```

Or for development:

```bash
cd apps/jira-status-update
uv sync
```

## Configuration

### Environment variables (required)

| Variable | Description |
|----------|-------------|
| `JIRA_USERNAME` | Atlassian account email |
| `JIRA_API_TOKEN` | [Atlassian API token](https://id.atlassian.com/manage-profile/security/api-tokens) |
| `GITHUB_TOKEN` | GitHub personal access token |
| `ANTHROPIC_API_KEY` | Anthropic API key (direct API) |

For Vertex AI instead of the direct API, set `ANTHROPIC_VERTEX_PROJECT_ID` (and optionally `CLOUD_ML_REGION`, default `us-east5`).

### Preferences file

Copy the default config and customize:

```bash
mkdir -p ~/.config/jira-status-update
cp apps/jira-status-update/prefs.default.toml ~/.config/jira-status-update/prefs.toml
```

Key settings:

```toml
[llm]
model = "claude-sonnet-5"
# coding_agent = "claude"   # enables tool-assisted revisions via Claude Code
writing_rules = [
  "No fractions or ratios like 7/8 or 4/25",
  "No completion percentages",
]

[jira]
url = "https://redhat.atlassian.net"
# skills_dir = "~/src/ai-helpers/plugins/jira/skills/status-analysis"

[profiles.my-team]
project = "MYPROJ"
component = "My Component"
label = "my-label"
exclude = ["bot@example.com"]
```

## Usage

```bash
# Basic usage
jira-status-update --project OCPSTRAT --component "Hosted Control Planes"

# Using a saved profile
jira-status-update --project OCPSTRAT --profile my-team

# Force re-gather (ignore cache)
jira-status-update --project OCPSTRAT --refresh

# Skip issues with no recent activity
jira-status-update --project OCPSTRAT --skip-quiet

# Custom date range
jira-status-update --project OCPSTRAT --start-date 2026-08-01 --end-date 2026-08-15
```

## TUI keybindings

The app has two phases: a pipeline screen (gather + draft) and the review screen.

### Review screen

| Key | Action |
|-----|--------|
| `a` | Approve draft and push to Jira |
| `s` | Skip issue |
| `e` | Edit draft in `$EDITOR` |
| `i` | Request AI revision (type instruction, press Enter) |
| `r` | Refresh current issue (re-gather + re-draft) |
| `R` | Refresh all issues (re-gather + re-draft) |
| `q` | Quit review |

When `coding_agent` is configured in prefs, `i` suspends the TUI and opens the agent interactively. Otherwise it uses a single API call.

## Running tests

```bash
cd apps/jira-status-update
uv run pytest tests/ -v
```

## Project structure

```
src/jira_status_update/
├── cli.py        # Argument parsing and entry point
├── config.py     # UserPrefs, path resolution, caching
├── llm.py        # LLM client, cost tracking, drafting, system prompt
├── gather.py     # Data gathering subprocess, issue processing, batch drafting
├── jira.py       # Jira API: auth, ADF conversion, field updates
├── agent.py      # Coding agent (Claude Code / OpenCode) integration
└── app.py        # Textual TUI: ReviewScreen and StatusUpdateApp
```

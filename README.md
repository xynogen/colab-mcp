# Colab MCP (Enhanced Fork)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](pyproject.toml)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple.svg)](https://modelcontextprotocol.io)

An MCP server for controlling Google Colab from any AI coding agent. Forked from
[`SebastianGilPinzon/colab-mcp`](https://github.com/SebastianGilPinzon/colab-mcp) (itself a
fork of the official [`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp)):
Sebastian fixed the bugs that block day-to-day use and restored the GPU control Google
removed; this fork adds a non-blocking cell runner plus runtime status/stop tools.

## Why this fork

The official `googlecolab/colab-mcp` has three blockers this fork solves:

1. **Invisible tools** ([#67](https://github.com/googlecolab/colab-mcp/discussions/67), [#69](https://github.com/googlecolab/colab-mcp/discussions/69)) — notebook tools rely on `notifications/tools/list_changed`, which most clients (Claude Code, Codex, Kiro) ignore, so only `open_colab_browser_connection` shows up. This fork pre-registers all tools at startup.
2. **"Disconnected from the local Colab MCP server"** ([#84](https://github.com/googlecolab/colab-mcp/discussions/84)) — root cause is an IPv4/IPv6 dual-stack bind (two sockets, different ports, tab reaches the wrong one). Fixed by forcing IPv4-only, plus a process registry (`--list-running`/`--kill-stale`) for orphaned servers.
3. **No programmatic GPU control** — Google [removed](https://github.com/googlecolab/colab-mcp/discussions/41) it. Restored via OAuth (`runtime_change`).

## Available Tools

| Tool | Browser | OAuth | Description |
|------|:---:|:---:|-------------|
| `open_colab_browser_connection` | ✓ | | Connect to Colab. Optional `notebook_url` opens a specific notebook (Drive URL, `#scrollTo=` preserved); empty creates a fresh notebook |
| `cell_add_code` | ✓ | | Add a code cell → returns the new `cellId`. Optional `afterCellId` places it after a known cell |
| `cell_add_text` | ✓ | | Add a markdown cell |
| `cell_run` | ✓ | | Execute a cell by `cellId` — **blocks** until it finishes |
| `cell_update` | ✓ | | Edit a cell by `cellId` |
| `cell_delete` | ✓ | | Delete a cell by `cellId` |
| `cell_move` | ✓ | | Move a cell — id-relative via `afterCellId` (or legacy `cellIndex`) |
| `cells_get` | ✓ | | Read notebook state (cell IDs, contents, outputs) |
| `cells_run` | ✓ | | Run a list of `cellIds` in order, **non-blocking** → returns a `jobId` |
| `cells_run_all` | ✓ | | Run every code cell in order, **non-blocking** → returns a `jobId` |
| `cells_run_status` | ✓ | | Poll a `jobId` for status + per-cell output (safe to poll; reads server memory) |
| `runtime_change` | | ✓ | Assign accelerator: T4/L4/A100 (GPU), V2-8/V5E-1/V6E-1 (TPU), or NONE |
| `runtime_status` | | ✓ | Report subscription tier, credit balance, hourly burn rate, assigned VMs |
| `runtime_stop` | | ✓ | Unassign all VMs to stop credit consumption |

Tools are grouped by object: `cell_*` (single cell), `cells_*` (whole-notebook / batch),
`runtime_*` (GPU/TPU). **For long cells (training, installs)** use `cells_run`/`cells_run_all` —
they return a `jobId` immediately instead of freezing the agent for the whole run; poll
`cells_run_status(jobId)`.

## Quick Start

Install [uv](https://docs.astral.sh/uv/) (do **not** `pip install uv` — it lacks features):

```bash
# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Add to your MCP client config (Claude Code, Cursor, etc.). Run from a clone:

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/colab-mcp", "colab-mcp"]
    }
  }
}
```

…or straight from git (no clone):

```json
{
  "mcpServers": {
    "colab-mcp": { "command": "uvx", "args": ["git+https://github.com/xynogen/colab-mcp"] }
  }
}
```

Restart your editor. All tools appear immediately. Call `open_colab_browser_connection`
(a Colab tab opens — accept Chrome's **Local Network Access** prompt if shown), then use
`cell_add_code`, `cell_run`, `cells_get`, etc.

## GPU / TPU Control (OAuth)

Enables `runtime_change` / `runtime_status` / `runtime_stop`. One-time setup:

1. Create a GCP project and [configure the OAuth consent screen](https://console.cloud.google.com/apis/credentials/consent) ("External"), adding your Google email as a **test user**.
2. [Create Credentials](https://console.cloud.google.com/apis/credentials) → OAuth client ID → **Desktop app** → download the JSON (e.g. `~/.config/colab-oauth.json`). (Client IDs can only be created in the web console.)
3. Add `--client-oauth-config` to your config args:

```json
"args": ["run", "--directory", "/path/to/colab-mcp", "colab-mcp",
         "--client-oauth-config", "/path/to/colab-oauth.json"]
```

First start opens the browser for consent; the token caches to `~/.colab-mcp-auth-token.json`
and auto-refreshes. Example flow:

```
runtime_change(accelerator="T4")     → Runtime changed to T4. Endpoint: gpu-t4-s-xxx
open_colab_browser_connection()      → Connection successful. Available notebook tools: …
cell_add_code(code="!nvidia-smi")    → {"newCellId": "abc123"}
cell_run(cellId="abc123")            → Tesla T4, 15GB …
runtime_stop()                       → Stopped 1 runtime(s)
```

## CLI Reference

| Flag | Description |
|------|-------------|
| _(none)_ | Start the MCP server (JSON-RPC on stdin/stdout) |
| `-l DIR`, `--log DIR` | Log directory (default: temp dir under `$TMPDIR`/`%TEMP%`) |
| `--client-oauth-config PATH` | OAuth client-secrets JSON; enables the `runtime_*` tools |
| `--list-running` | List running `colab-mcp` servers (pid, port, host, start time) and exit |
| `--kill-stale` | Terminate all running servers + clear the registry, then exit |

Run `--list-running` / `--kill-stale` from a **regular shell**, not from inside your agent
(which is itself an MCP instance). The server keeps a registry at `~/.colab-mcp/registry.json`
(`%LOCALAPPDATA%\colab-mcp\registry.json` on Windows), pruning dead entries on startup.

## Troubleshooting

**Tools don't appear** — you're on this fork (not the official repo); define the server in
only ONE config file; restart your editor.

**`runtime_*` says "Runtime API not initialized"** — `--client-oauth-config` missing or the
JSON path is wrong. Check the latest log (`ls -t $TMPDIR/colab-mcp-logs-*/colab-mcp.*.log | head -1 | xargs cat`) for `INFO:Colab API client ready`.

**OAuth "Access denied"** — add your Google email as a test user on the consent screen.

**Browser opens but times out** — this is almost always Chrome's **Local Network Access** policy.
A public site (colab.research.google.com) talking to `ws://localhost` is blocked by default:

- On the prompt, click **Allow**. If you previously clicked **Block**, Chrome remembers it per-site and silently cancels every attempt. Reset via the lock icon → _Reset permissions_, or `chrome://settings/content/siteDetails?site=https%3A%2F%2Fcolab.research.google.com` → set "Access other devices on the network" to **Ask**. Edge/Firefox are unaffected.

**Stale tab / "Disconnected"** — an old Colab tab points at a dead port. Close every
`colab.research.google.com` tab and retry (each server uses a unique `?p=<port>` URL so
Chrome opens a fresh tab). If it persists, run `colab-mcp --kill-stale` from a shell to clear
orphaned servers, then reconnect.

**Windows port blocked (WinError 10013)** — fixed in this fork (port 8085). If it recurs,
change `OAUTH_SERVER_PORT` in `src/colab_mcp/auth.py`.

## Development

```bash
mise run check   # lint + test  (needs mise; else: uv run ruff check . && uv run pytest)
mise run test    # pytest
mise run cov     # pytest + coverage
```

## Changes from upstream

Fixes: pre-register all tools at startup (invisible-tools, [#67](https://github.com/googlecolab/colab-mcp/discussions/67)/[#69](https://github.com/googlecolab/colab-mcp/discussions/69)) ·
restore `runtime_change` GPU control via OAuth · fix `ColabClient` init + Windows OAuth port ·
match real Colab API signatures · IPv4-only bind + Private Network Access headers (real cause of
"Disconnected", [#84](https://github.com/googlecolab/colab-mcp/discussions/84)) · stale-server registry with `--list-running`/`--kill-stale`.

Later additions in this fork:

- **`notebook_url`** on `open_colab_browser_connection` — open a specific notebook, or create a fresh one when empty.
- **Non-blocking runner** — `cells_run` / `cells_run_all` / `cells_run_status`, so long cells don't freeze the agent.
- **`runtime_status` + `runtime_stop`** — expose credit balance, burn rate, assigned VMs, and an off switch.
- **`runtime_*` rename** (`change_runtime`→`runtime_change`, `stop_runtime`→`runtime_stop`) + TPU variant fix.
- **id-based placement** — `afterCellId` on `cell_add_code`/`cell_move` instead of integer positions.
- **`cell_*` / `cells_*` / `runtime_*` naming** — tools grouped by object for a cleaner surface.
- **Tooling** — `pyrightconfig.json`, `mise.toml`, and a 67-test suite (~65% coverage).

Google [does not accept external contributions](https://github.com/googlecolab/colab-mcp/blob/main/CONTRIBUTING.md), so these live here.

## License

Apache License 2.0 (same as upstream). Fork chain:
[`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp) →
[`SebastianGilPinzon/colab-mcp`](https://github.com/SebastianGilPinzon/colab-mcp) →
[`xynogen/colab-mcp`](https://github.com/xynogen/colab-mcp).

Copyright 2026 Google Inc. (original) · Sebastian Gil Pinzón (fork fixes) · Muhammad Fikri
(xynogen, this fork). All distributed under Apache 2.0 — the original copyright and license
notices are retained in every source file per §4 of the license. See [LICENSE](LICENSE).

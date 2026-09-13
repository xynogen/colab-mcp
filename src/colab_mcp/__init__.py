# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
import tempfile
import webbrowser

from fastmcp import FastMCP
from fastmcp.utilities import logging as fastmcp_logger

from colab_mcp import process_registry
from colab_mcp.session import NOT_CONNECTED_MSG, ColabSessionProxy
from colab_mcp.websocket_server import COLAB

mcp = FastMCP(name="ColabMCP")

# These will be set during main_async() startup
_proxy_client = None
_session_mcp = None
_colab_client = None  # For runtime API (assign/unassign GPU)

# In-memory registry of async cell-run jobs. run_cells() spawns a background
# task (holding the single browser websocket for the whole run) and returns a
# jobId immediately; get_run_status() reads this dict, never the browser, so
# polling is instant and decoupled from the blocking run_code_cell round-trip.
_run_jobs: dict = {}
_run_job_seq = 0

# Guards against a second open_colab_browser_connection call spawning another
# browser tab while the first is still waiting for the tab to connect back.
_connect_in_flight = False


async def _run_cells_job(job_id: str, cell_ids: list) -> None:
    """Background worker: run each cell in order, record output/errors into the job."""
    job = _run_jobs[job_id]
    job["status"] = "running"
    for cid in cell_ids:
        job["current"] = cid
        out = await _forward_or_stub("run_code_cell", {"cellId": cid})
        job["results"].append({"cellId": cid, "output": out})
        if out == NOT_CONNECTED_MSG or out.startswith("Error calling"):
            job["status"] = "error"
            job["current"] = None
            return
    job["status"] = "done"
    job["current"] = None


async def _forward_or_stub(tool_name: str, arguments: dict) -> str:
    """Forward a tool call to the browser if connected, otherwise return stub message."""
    if _proxy_client is not None and _proxy_client.is_connected():
        assert (
            _proxy_client.proxy_mcp_client is not None
        )  # is_connected() guarantees this
        try:
            result = await _proxy_client.proxy_mcp_client.call_tool(
                tool_name, arguments
            )
            # Extract text from result
            if hasattr(result, "content"):
                return "\n".join(
                    t for c in result.content if (t := getattr(c, "text", None))
                )
            return str(result)
        except Exception as e:
            return f"Error calling {tool_name}: {e}. Try calling open_colab_browser_connection to reconnect."
    return NOT_CONNECTED_MSG


def _build_colab_url(notebook_url: str, wss) -> str:
    """Merge a Colab notebook URL with the MCP proxy token/port.

    `p=<port>` in the query forces a unique URL per server instance so Chrome can't
    silently reuse a stale tab (whose fragment points at a dead port). The token/port
    live in the fragment, which Colab's browser-side code reads as the source of truth.
    A user-supplied URL's own query/fragment (e.g. #scrollTo=...) is preserved.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    # Empty URL -> create a fresh untitled notebook (#create=true), not the
    # shared scratch empty.ipynb.
    base = notebook_url.strip() or f"{COLAB}/#create=true"
    parts = urlsplit(base)

    query = parse_qsl(parts.query, keep_blank_values=True)
    query.append(("p", str(wss.port)))

    # Fragment is &-joined key=value pairs; append ours after any existing ones.
    frag = parts.fragment
    proxy_frag = f"mcpProxyToken={wss.token}&mcpProxyPort={wss.port}"
    fragment = f"{frag}&{proxy_frag}" if frag else proxy_frag

    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), fragment)
    )


def _default_browser_hint() -> str:
    """Best-effort name of the OS default browser, so a timeout message can point
    the user at the right window (the tab opens in the DEFAULT browser, which may
    not be the one they're looking at)."""
    if os.environ.get("BROWSER"):
        return os.environ["BROWSER"]
    try:
        import subprocess

        out = subprocess.run(
            ["xdg-settings", "get", "default-web-browser"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        return out or "your default browser"
    except Exception:
        return "your default browser"


@mcp.tool()
async def open_colab_browser_connection(notebook_url: str = "") -> str:
    """Opens a connection to a Google Colab browser session and unlocks notebook editing tools.

    Pass a full Colab notebook URL (e.g. https://colab.research.google.com/drive/<id>) to
    open that notebook; leave empty to create a fresh new notebook. Any query/fragment on
    the URL is preserved. Returns whether the connection attempt succeeded."""
    global _connect_in_flight
    # Fall back to COLAB_MCP_NOTEBOOK_URL when no URL is passed, so a server can
    # be pinned to a specific notebook without the caller repeating it. Empty
    # both -> a fresh untitled notebook (handled downstream by _build_colab_url).
    notebook_url = notebook_url.strip() or os.environ.get("COLAB_MCP_NOTEBOOK_URL", "")
    if _proxy_client is not None and _proxy_client.is_connected():
        return "Already connected to Colab."

    if _proxy_client is None:
        return "Server not initialized. Please wait and try again."

    # #2 Reuse tab in-flight: a second call while the first is still waiting
    # would spawn a duplicate tab that steals focus and confuses the user.
    if _connect_in_flight:
        return (
            "A connection attempt is already in progress — a Colab tab was just "
            "opened and is waiting to connect. Check your browser (and accept any "
            "Local Network Access prompt) instead of opening another tab."
        )

    # #5 Self-heal: drop dead peer servers from the registry before opening a
    # tab, so timeout diagnostics don't blame orphans that no longer exist.
    try:
        process_registry.prune_dead()
    except Exception:
        pass

    browser = _default_browser_hint()
    logging.info(
        f"Opening Colab tab in {browser}; waiting up to 60s for it to connect "
        f"back to ws://127.0.0.1:{_proxy_client.wss.port}"
    )
    _connect_in_flight = True
    try:
        webbrowser.open_new(_build_colab_url(notebook_url, _proxy_client.wss))
        # Wait for browser to connect
        await _proxy_client.await_proxy_connection()
    finally:
        _connect_in_flight = False

    if _proxy_client.is_connected():
        tool_names = await _proxy_client.await_tools_ready()
        tools_text = ", ".join(tool_names) if tool_names else "none discovered"
        return f"Connection successful. Available notebook tools: {tools_text}. You can now create, edit, and execute cells in the Colab notebook."

    # Timed out — surface diagnostic info about other running servers so the
    # user can recognize the "old browser tab pointed at a dead port" case.
    try:
        others = [e for e in process_registry.list_running() if e.pid != os.getpid()]
    except Exception:
        others = []
    my_port = _proxy_client.wss.port
    if others:
        peer_ports = ", ".join(f"{e.port} (pid {e.pid})" for e in others)
        return (
            f"Connection timed out (tab opened in {browser}). This server is on "
            f"port {my_port}, but {len(others)} other colab-mcp server(s) are also "
            f"running: {peer_ports}. If you have an old Colab tab open, it may be "
            "pointing at one of those instead of this server. Either close "
            "the old tab and let me open a fresh one, or run `colab-mcp "
            "--kill-stale` to clean up orphaned servers."
        )
    return (
        f"Connection timed out. The tab was opened in {browser} (port {my_port}) "
        "— make sure you're looking at THAT browser, not another one. Common causes:\n"
        "  1. Wrong/hidden browser - the tab opens in your OS default browser. If "
        f"that's not where you're looking, either switch to {browser}, or set "
        "BROWSER=<name> in the MCP server env and restart.\n"
        "  2. Stale Colab tab - an old tab whose URL points at a dead port shows "
        "'Disconnected'. Close every colab.research.google.com tab, then retry "
        "(each server uses a unique `?p=<port>` URL to force a fresh tab).\n"
        "  3. Local Network Access blocked (Chrome only) - click 'Allow' on the "
        "prompt; if you clicked 'Block' before, reset it in site settings. "
        "Firefox/Edge/Zen are unaffected."
    )


async def _index_after(after_cell_id: str) -> int | None:
    """Resolve an afterCellId to the insert/move index just after it.

    Lets tools take a cellId (the handle the model already has) instead of a
    fragile integer position. Returns None if the id isn't found. The browser
    handlers only understand cellIndex, so we translate here via get_cells.
    """
    raw = await _forward_or_stub("get_cells", {})
    try:
        cells = json.loads(raw).get("cells", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    for i, c in enumerate(cells):
        if c.get("id") == after_cell_id:
            return i + 1
    return None


@mcp.tool()
async def cell_add_code(
    code: str = "",
    after_cell_id: str = "",
    language: str = "python",
    cell_index: int = 0,
) -> str:
    """Add a new code cell. Returns the new cell_id. By default appends after after_cell_id
    (the cell_id to insert after, from add_code_cell/get_cells); omit it to insert at the
    top. cell_index is a legacy positional fallback. Requires an active browser connection."""
    if after_cell_id:
        idx = await _index_after(after_cell_id)
        if idx is None:
            return f"No such cell_id: {after_cell_id}"
        cell_index = idx
    return await _forward_or_stub(
        "add_code_cell", {"cellIndex": cell_index, "code": code, "language": language}
    )


@mcp.tool()
async def cell_add_text(content: str = "", cell_index: int = -1) -> str:
    """Add a new text/markdown cell to the Colab notebook. Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub(
        "add_text_cell", {"content": content, "cellIndex": cell_index}
    )


@mcp.tool()
async def cells_get() -> str:
    """Read the current notebook state: list of cells with their IDs, contents, and outputs. Essential for iterative work (write -> run -> read -> adjust). Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub("get_cells", {})


@mcp.tool()
async def cell_run(cell_id: str = "") -> str:
    """Execute a code cell in the Colab notebook by cell_id (from add_code_cell or get_cells). Blocks until the cell finishes. Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub("run_code_cell", {"cellId": cell_id})


@mcp.tool()
async def cells_run(cell_ids: list[str]) -> str:
    """Run one or more cells in order WITHOUT blocking. Returns a job_id immediately;
    poll cells_run_status(job_id) for progress and per-cell output. Use this for
    long-running cells (training, installs) so the agent isn't frozen for the whole
    run. Requires an active browser connection via open_colab_browser_connection."""
    global _run_job_seq
    if _proxy_client is None or not _proxy_client.is_connected():
        return NOT_CONNECTED_MSG
    if not cell_ids:
        return "No cell_ids provided."
    _run_job_seq += 1
    job_id = f"run-{_run_job_seq}"
    _run_jobs[job_id] = {
        "status": "pending",
        "cellIds": list(cell_ids),
        "current": None,
        "results": [],
    }
    asyncio.create_task(_run_cells_job(job_id, list(cell_ids)))
    return json.dumps({"jobId": job_id, "status": "pending", "cellIds": list(cell_ids)})


@mcp.tool()
async def cells_run_status(job_id: str = "") -> str:
    """Read the status and captured output of an async run started by cells_run.
    Returns status (pending|running|done|error), the currently-running cell_id, and
    per-cell outputs collected so far. Reads server memory only — safe to poll."""
    job = _run_jobs.get(job_id)
    if job is None:
        return json.dumps({"error": f"No such job_id: {job_id}"})
    return json.dumps({"jobId": job_id, **job})


@mcp.tool()
async def cells_run_all() -> str:
    """Run every code cell in the notebook in order, WITHOUT blocking. Reads the current
    cell list, then starts an async job like cells_run. Returns a jobId immediately;
    poll cells_run_status(jobId) for progress. Requires an active browser connection."""
    global _run_job_seq
    if _proxy_client is None or not _proxy_client.is_connected():
        return NOT_CONNECTED_MSG
    raw = await _forward_or_stub("get_cells", {})
    try:
        cells = json.loads(raw).get("cells", [])
    except (json.JSONDecodeError, AttributeError):
        return f"Could not read cells: {raw}"
    cell_ids = [c["id"] for c in cells if c.get("cell_type") == "code" and c.get("id")]
    if not cell_ids:
        return "No code cells to run."
    _run_job_seq += 1
    job_id = f"run-{_run_job_seq}"
    _run_jobs[job_id] = {
        "status": "pending",
        "cellIds": cell_ids,
        "current": None,
        "results": [],
    }
    asyncio.create_task(_run_cells_job(job_id, cell_ids))
    return json.dumps({"jobId": job_id, "status": "pending", "cellIds": cell_ids})


@mcp.tool()
async def cell_update(cell_id: str = "", content: str = "") -> str:
    """Update the contents of an existing cell in the Colab notebook. Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub(
        "update_cell", {"cellId": cell_id, "content": content}
    )


@mcp.tool()
async def cell_delete(cell_id: str = "") -> str:
    """Delete a cell from the Colab notebook by cell_id. Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub("delete_cell", {"cellId": cell_id})


@mcp.tool()
async def cell_move(
    cell_id: str = "", after_cell_id: str = "", cell_index: int = 0
) -> str:
    """Move a cell (by cell_id) to just after after_cell_id (another cell_id). This is the
    id-only way to reorder — no index counting. cell_index is a legacy positional
    fallback. Requires an active browser connection."""
    if after_cell_id:
        idx = await _index_after(after_cell_id)
        if idx is None:
            return f"No such cell_id: {after_cell_id}"
        cell_index = idx
    return await _forward_or_stub(
        "move_cell", {"cellId": cell_id, "cellIndex": cell_index}
    )


@mcp.tool()
async def runtime_change(accelerator: str = "T4") -> str:
    """Change the Colab runtime accelerator. Valid values: NONE, T4, L4, A100 (GPU) and
    V2-8, V5E-1, V6E-1 (TPU). Requires OAuth setup (first time opens browser for consent)."""
    if _colab_client is None:
        return "Runtime API not initialized. Start with --client-oauth-config flag pointing to your OAuth client secrets JSON."
    try:
        import uuid

        from colab_mcp.client import Accelerator, Variant

        acc = Accelerator(accelerator)
        if acc == Accelerator.NONE:
            variant = Variant.DEFAULT
        elif acc.value.startswith("V"):  # V2-8 / V5E-1 / V6E-1 are TPUs
            variant = Variant.TPU
        else:
            variant = Variant.GPU
        notebook_hash = uuid.uuid4()

        # Unassign current VM if any
        try:
            assignments = _colab_client.list_assignments()
            for a in assignments:
                _colab_client.unassign(a.endpoint)
        except Exception:
            pass

        # Assign new VM. assign() returns either a PostAssignmentResponse
        # (has .endpoint) or a dict {"assignment": Assignment} when a VM was
        # already assigned — unwrap both.
        result = _colab_client.assign(notebook_hash, variant, acc)
        endpoint = getattr(result, "endpoint", None)
        if endpoint is None and isinstance(result, dict):
            assignment = result.get("assignment")
            endpoint = getattr(assignment, "endpoint", None)
        return f"Runtime changed to {accelerator}. Endpoint: {endpoint}. Use open_colab_browser_connection to connect to the new runtime."
    except Exception as e:
        return f"Failed to change runtime: {e}"


@mcp.tool()
async def runtime_status() -> str:
    """Report Colab account + runtime state in one call: subscription tier, credit
    balance, hourly burn rate, and every currently-assigned VM (accelerator + endpoint).
    Use to check whether a GPU/TPU is attached and burning credits. Requires OAuth setup."""
    if _colab_client is None:
        return "Runtime API not initialized. Start with --client-oauth-config flag pointing to your OAuth client secrets JSON."
    status: dict = {}
    try:
        status["subscription_tier"] = _colab_client.get_subscription_tier().name
    except Exception as e:
        status["subscription_tier"] = f"error: {e}"
    try:
        ccu = _colab_client.get_ccu_info()
        status["credit_balance"] = ccu.current_balance
        status["consumption_rate_hourly"] = ccu.consumption_rate_hourly
        status["assignments_count"] = ccu.assignments_count
    except Exception as e:
        status["ccu"] = f"error: {e}"
    try:
        status["assignments"] = [
            {"accelerator": a.accelerator.value, "endpoint": a.endpoint}
            for a in _colab_client.list_assignments()
        ]
    except Exception as e:
        status["assignments"] = f"error: {e}"
    return json.dumps(status)


@mcp.tool()
async def runtime_stop() -> str:
    """Unassign (stop) all currently-assigned Colab VMs to halt credit consumption.
    The off switch for runtime_change. Requires OAuth setup."""
    if _colab_client is None:
        return "Runtime API not initialized. Start with --client-oauth-config flag pointing to your OAuth client secrets JSON."
    try:
        assignments = _colab_client.list_assignments()
        if not assignments:
            return "No runtimes currently assigned."
        stopped = []
        for a in assignments:
            _colab_client.unassign(a.endpoint)
            stopped.append(a.endpoint)
        return f"Stopped {len(stopped)} runtime(s): {', '.join(stopped)}"
    except Exception as e:
        return f"Failed to stop runtime: {e}"


def init_logger(logdir):
    log_filename = datetime.datetime.now().strftime(
        f"{logdir}/colab-mcp.%Y-%m-%d_%H-%M-%S.log"
    )
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_filename,
        level=logging.INFO,
    )
    fastmcp_logger.get_logger("colab-mcp").info("logging to %s" % log_filename)


def parse_args(v):
    parser = argparse.ArgumentParser(
        description="ColabMCP is an MCP server that lets you interact with Colab."
    )
    parser.add_argument(
        "-l",
        "--log",
        help="if set, use this directory as a location for logfiles (if unset, will log to %s/colab-mcp-logs/)"
        % tempfile.gettempdir(),
        action="store",
        default=tempfile.mkdtemp(prefix="colab-mcp-logs-"),
    )
    parser.add_argument(
        "-p",
        "--enable-proxy",
        help="if set, enable the runtime proxy (enabled by default).",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--client-oauth-config",
        help="Path to OAuth client secrets JSON for Colab API access (enables runtime_change tool).",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--port",
        help="Bind the browser websocket to this fixed port instead of a random "
        "one. Lets a restarted server re-adopt a stale Colab tab on refresh. "
        "Overrides COLAB_MCP_PORT.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--token",
        help="Use this fixed proxy token instead of a random one. Pair with "
        "--port so a stale tab's mcpProxyToken still authorizes. Overrides "
        "COLAB_MCP_TOKEN.",
        default=None,
    )
    parser.add_argument(
        "--notebook-url",
        help="Default Colab notebook URL to open when the connect tool is called "
        "without one. Overrides COLAB_MCP_NOTEBOOK_URL.",
        default=None,
    )
    parser.add_argument(
        "--list-running",
        help="List all currently-running colab-mcp servers and exit.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--kill-stale",
        help="Terminate all running colab-mcp servers (including this one is NOT included) and exit. Useful when the browser shows 'Disconnected from the local Colab MCP server' due to orphaned processes from prior sessions.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--reset-identity",
        help="Delete the persisted port+token (~/.colab-mcp/identity.json) so the "
        "next server run gets a fresh random port+token, then exit.",
        action="store_true",
        default=False,
    )
    return parser.parse_args(v)


def _print_running_servers() -> None:
    entries = process_registry.list_running()
    if not entries:
        print("No colab-mcp servers currently registered as running.")
        return
    print(f"Found {len(entries)} running colab-mcp server(s):")
    import datetime as _dt

    for e in entries:
        started = _dt.datetime.fromtimestamp(e.started_at).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  pid={e.pid:<6}  port={e.port:<6}  host={e.host}  started={started}")


async def main_async():
    global _proxy_client, _session_mcp, _colab_client
    args = parse_args(sys.argv[1:])
    init_logger(args.log)

    # CLI flags win over env; the websocket server reads these env vars when it
    # constructs, so setting them here keeps env as the single source of truth.
    if args.port is not None:
        os.environ["COLAB_MCP_PORT"] = str(args.port)
    if args.token is not None:
        os.environ["COLAB_MCP_TOKEN"] = args.token
    if args.notebook_url is not None:
        os.environ["COLAB_MCP_NOTEBOOK_URL"] = args.notebook_url

    # Diagnostic / cleanup flags exit early.
    if args.list_running:
        _print_running_servers()
        return
    if args.kill_stale:
        removed = process_registry.cleanup_stale(kill=True)
        if not removed:
            print("No stale colab-mcp servers found.")
        else:
            print(f"Terminated {len(removed)} stale colab-mcp server(s):")
            for e in removed:
                print(f"  pid={e.pid} port={e.port}")
        return
    if args.reset_identity:
        if process_registry.clear_identity():
            print("Cleared persisted identity; next run gets a fresh port+token.")
        else:
            print("No persisted identity to clear.")
        return

    # Prune any dead entries from prior crashed runs BEFORE we bind a port.
    # This keeps the registry honest. We don't auto-kill ALIVE entries here —
    # multiple clients (e.g., two Claude Code instances) are valid; only the
    # browser-tab confusion is the bug, and the per-tab token fragment scopes
    # which server a tab talks to.
    dead = process_registry.prune_dead()
    if dead:
        logging.info(f"Pruned {dead} stale entries from process registry")

    if args.enable_proxy:
        logging.info("enabling session proxy tools")
        _session_mcp = ColabSessionProxy()
        await _session_mcp.start_proxy_server()
        assert _session_mcp.wss is not None  # set by start_proxy_server()
        _proxy_client = _session_mcp.proxy_client
        # Register ourselves now that we know the port.
        try:
            entry = process_registry.register(
                port=_session_mcp.wss.port,
                host=_session_mcp.wss.host,
            )
            logging.info(f"Registered colab-mcp pid={entry.pid} port={entry.port}")
        except Exception as exc:
            logging.warning(f"Could not register process: {exc}")

    if args.client_oauth_config:
        try:
            from colab_mcp.auth import get_credentials
            from colab_mcp.client import ColabClient, Prod

            logging.info("initializing Colab API client with OAuth")
            session = get_credentials(args.client_oauth_config)
            _colab_client = ColabClient(Prod(), session)
            logging.info("Colab API client ready")
        except Exception as e:
            logging.warning(f"Failed to initialize Colab API client: {e}")

    try:
        await mcp.run_async()

    finally:
        if args.enable_proxy and _session_mcp:
            await _session_mcp.cleanup()
        # Always unregister so a clean shutdown doesn't leave a stale entry.
        try:
            process_registry.unregister()
        except Exception as exc:
            logging.warning(f"Could not unregister process: {exc}")


def main() -> None:
    asyncio.run(main_async())

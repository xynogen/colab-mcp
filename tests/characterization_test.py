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

"""Characterization tests: pin the CURRENT observable behavior so an accidental
change to a message, flag default, or branch shows up as a failing test. These
assert what the code does today, not what it ideally should do."""

from types import SimpleNamespace

import pytest

import colab_mcp
from colab_mcp import parse_args

# --- parse_args: flag defaults and toggles ---


def test_parse_args_defaults():
    args = parse_args([])
    assert args.enable_proxy is True
    assert args.client_oauth_config is None
    assert args.list_running is False
    assert args.kill_stale is False
    assert args.log  # a temp dir is allocated by default


def test_parse_args_flags():
    args = parse_args(["--list-running", "--kill-stale"])
    assert args.list_running is True
    assert args.kill_stale is True


def test_parse_args_oauth_config():
    args = parse_args(["--client-oauth-config", "/tmp/secrets.json"])
    assert args.client_oauth_config == "/tmp/secrets.json"


# --- notebook tools: not-connected stub messages ---


@pytest.fixture
def disconnected(monkeypatch):
    monkeypatch.setattr(colab_mcp, "_proxy_client", None)


@pytest.mark.asyncio
async def test_notebook_tools_not_connected(disconnected):
    msg = colab_mcp.NOT_CONNECTED_MSG
    assert await colab_mcp.cell_add_code.fn(code="x") == msg
    assert await colab_mcp.cell_add_text.fn(content="x") == msg
    assert await colab_mcp.cells_get.fn() == msg
    assert await colab_mcp.cell_run.fn(cellId="c") == msg
    assert await colab_mcp.cell_update.fn(cellId="c", content="x") == msg
    assert await colab_mcp.cell_delete.fn(cellId="c") == msg
    assert await colab_mcp.cell_move.fn(cellId="c") == msg


@pytest.mark.asyncio
async def test_run_cells_not_connected_message(disconnected):
    assert await colab_mcp.cells_run.fn(["c"]) == colab_mcp.NOT_CONNECTED_MSG
    assert await colab_mcp.cells_run_all.fn() == colab_mcp.NOT_CONNECTED_MSG


# --- open_colab_browser_connection: guard branches (no browser) ---


def test_default_browser_hint_respects_env(monkeypatch):
    monkeypatch.setenv("BROWSER", "firefox")
    assert colab_mcp._default_browser_hint() == "firefox"


def test_wss_port_token_default_random(monkeypatch, tmp_path):
    from colab_mcp import process_registry
    from colab_mcp.websocket_server import ColabWebSocketServer

    monkeypatch.delenv("COLAB_MCP_PORT", raising=False)
    monkeypatch.delenv("COLAB_MCP_TOKEN", raising=False)
    # No persisted identity -> random.
    monkeypatch.setattr(process_registry, "_registry_dir", lambda: tmp_path)
    w = ColabWebSocketServer()
    assert w._bind_port == 0
    assert len(w.token) >= 16


def test_identity_roundtrip_and_clear(monkeypatch, tmp_path):
    from colab_mcp import process_registry

    monkeypatch.setattr(process_registry, "_registry_dir", lambda: tmp_path)
    assert process_registry.load_identity() == {}
    process_registry.save_identity(40123, "tok")
    assert process_registry.load_identity() == {"port": 40123, "token": "tok"}
    assert process_registry.clear_identity() is True
    assert process_registry.load_identity() == {}
    assert process_registry.clear_identity() is False


def test_wss_reuses_persisted_identity(monkeypatch, tmp_path):
    from colab_mcp import process_registry
    from colab_mcp.websocket_server import ColabWebSocketServer

    monkeypatch.delenv("COLAB_MCP_PORT", raising=False)
    monkeypatch.delenv("COLAB_MCP_TOKEN", raising=False)
    monkeypatch.setattr(process_registry, "_registry_dir", lambda: tmp_path)
    process_registry.save_identity(40123, "persisted-tok")
    w = ColabWebSocketServer()
    assert w._bind_port == 40123
    assert w.token == "persisted-tok"


def test_wss_port_token_env_override(monkeypatch):
    from colab_mcp.websocket_server import ColabWebSocketServer

    monkeypatch.setenv("COLAB_MCP_PORT", "35755")
    monkeypatch.setenv("COLAB_MCP_TOKEN", "reuseme")
    w = ColabWebSocketServer()
    assert w._bind_port == 35755
    assert w.token == "reuseme"


def test_cli_port_token_flags():
    a = colab_mcp.parse_args(["--port", "41000", "--token", "tk"])
    assert a.port == 41000
    assert a.token == "tk"
    b = colab_mcp.parse_args([])
    assert b.port is None and b.token is None


def test_cli_notebook_url_flag():
    a = colab_mcp.parse_args(["--notebook-url", "https://x/drive/abc"])
    assert a.notebook_url == "https://x/drive/abc"
    assert colab_mcp.parse_args([]).notebook_url is None


def test_cli_reset_identity_flag():
    assert colab_mcp.parse_args(["--reset-identity"]).reset_identity is True
    assert colab_mcp.parse_args([]).reset_identity is False


@pytest.mark.asyncio
async def test_open_connection_uses_env_notebook_url(monkeypatch):
    # No URL passed + env set -> the env URL is what gets opened.
    monkeypatch.setenv("COLAB_MCP_NOTEBOOK_URL", "https://col/drive/envnb")

    async def never(*_):
        return None

    opened = {}
    proxy = SimpleNamespace(
        is_connected=lambda: False,
        wss=SimpleNamespace(port=40000, token="tok"),
        await_proxy_connection=never,
    )
    monkeypatch.setattr(colab_mcp, "_proxy_client", proxy)
    monkeypatch.setattr(colab_mcp, "_connect_in_flight", False)
    monkeypatch.setattr(colab_mcp.process_registry, "prune_dead", lambda: 0)
    monkeypatch.setattr(colab_mcp.process_registry, "list_running", list)
    monkeypatch.setattr(
        colab_mcp.webbrowser, "open_new", lambda url: opened.setdefault("url", url)
    )
    await colab_mcp.open_colab_browser_connection.fn()
    assert "envnb" in opened["url"]


@pytest.mark.asyncio
async def test_open_connection_uninitialized(monkeypatch):
    monkeypatch.setattr(colab_mcp, "_proxy_client", None)
    out = await colab_mcp.open_colab_browser_connection.fn()
    assert out == "Server not initialized. Please wait and try again."


@pytest.mark.asyncio
async def test_open_connection_already_connected(monkeypatch):
    proxy = SimpleNamespace(is_connected=lambda: True)
    monkeypatch.setattr(colab_mcp, "_proxy_client", proxy)
    out = await colab_mcp.open_colab_browser_connection.fn()
    assert out == "Already connected to Colab."


@pytest.mark.asyncio
async def test_open_connection_rejects_second_in_flight(monkeypatch):
    # A connect is already waiting -> a second call must not open another tab.
    proxy = SimpleNamespace(
        is_connected=lambda: False,
        wss=SimpleNamespace(port=40000, token="tok"),
    )
    monkeypatch.setattr(colab_mcp, "_proxy_client", proxy)
    monkeypatch.setattr(colab_mcp, "_connect_in_flight", True)
    out = await colab_mcp.open_colab_browser_connection.fn()
    assert "already in progress" in out


@pytest.mark.asyncio
async def test_open_connection_timeout_reports_peers(monkeypatch):
    # Never connects; a peer server exists -> the "other servers running" branch.
    async def never(*_):
        return None

    proxy = SimpleNamespace(
        is_connected=lambda: False,
        wss=SimpleNamespace(port=40000, token="tok"),
        await_proxy_connection=never,
    )
    monkeypatch.setattr(colab_mcp, "_proxy_client", proxy)
    monkeypatch.setattr(colab_mcp.webbrowser, "open_new", lambda url: None)
    monkeypatch.setattr(
        colab_mcp.process_registry,
        "list_running",
        lambda: [SimpleNamespace(pid=999, port=50000, host="127.0.0.1")],
    )
    out = await colab_mcp.open_colab_browser_connection.fn()
    assert "Connection timed out" in out
    assert "50000 (pid 999)" in out
    assert "--kill-stale" in out


@pytest.mark.asyncio
async def test_open_connection_timeout_no_peers(monkeypatch):
    async def never(*_):
        return None

    proxy = SimpleNamespace(
        is_connected=lambda: False,
        wss=SimpleNamespace(port=40000, token="tok"),
        await_proxy_connection=never,
    )
    monkeypatch.setattr(colab_mcp, "_proxy_client", proxy)
    monkeypatch.setattr(colab_mcp.webbrowser, "open_new", lambda url: None)
    monkeypatch.setattr(colab_mcp.process_registry, "list_running", list)
    out = await colab_mcp.open_colab_browser_connection.fn()
    assert "Connection timed out" in out
    assert "Common causes" in out
    assert "Local Network Access" in out

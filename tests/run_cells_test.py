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

import asyncio
import json

import pytest

import colab_mcp


@pytest.fixture(autouse=True)
def reset_jobs():
    colab_mcp._run_jobs.clear()
    colab_mcp._run_job_seq = 0
    yield
    colab_mcp._run_jobs.clear()


def _connected_proxy():
    class _P:
        def is_connected(self):
            return True

    return _P()


async def _call(tool):
    """Invoke a FastMCP-wrapped tool's underlying function."""
    return tool.fn if hasattr(tool, "fn") else tool


@pytest.mark.asyncio
async def test_run_cells_not_connected(monkeypatch):
    monkeypatch.setattr(colab_mcp, "_proxy_client", None)
    out = await colab_mcp.run_cells.fn(["a"])
    assert out == colab_mcp.NOT_CONNECTED_MSG


@pytest.mark.asyncio
async def test_run_cells_empty_list(monkeypatch):
    monkeypatch.setattr(colab_mcp, "_proxy_client", _connected_proxy())
    out = await colab_mcp.run_cells.fn([])
    assert "No cellIds" in out


@pytest.mark.asyncio
async def test_run_cells_lifecycle_done(monkeypatch):
    monkeypatch.setattr(colab_mcp, "_proxy_client", _connected_proxy())

    async def fake_forward(tool_name, arguments):
        await asyncio.sleep(0.01)
        return f"output for {arguments['cellId']}"

    monkeypatch.setattr(colab_mcp, "_forward_or_stub", fake_forward)

    start = json.loads(await colab_mcp.run_cells.fn(["c1", "c2"]))
    assert start["status"] == "pending"
    job_id = start["jobId"]

    # Returned immediately, before work completed.
    mid = json.loads(await colab_mcp.get_run_status.fn(job_id))
    assert mid["status"] in ("pending", "running")

    # Let the background task finish.
    await asyncio.sleep(0.1)
    done = json.loads(await colab_mcp.get_run_status.fn(job_id))
    assert done["status"] == "done"
    assert [r["cellId"] for r in done["results"]] == ["c1", "c2"]
    assert done["results"][0]["output"] == "output for c1"
    assert done["current"] is None


@pytest.mark.asyncio
async def test_run_cells_stops_on_error(monkeypatch):
    monkeypatch.setattr(colab_mcp, "_proxy_client", _connected_proxy())

    async def fake_forward(tool_name, arguments):
        if arguments["cellId"] == "bad":
            return "Error calling run_code_cell: boom"
        return "ok"

    monkeypatch.setattr(colab_mcp, "_forward_or_stub", fake_forward)

    job_id = json.loads(await colab_mcp.run_cells.fn(["good", "bad", "never"]))["jobId"]
    await asyncio.sleep(0.05)
    st = json.loads(await colab_mcp.get_run_status.fn(job_id))
    assert st["status"] == "error"
    # Stopped at the failing cell; the third never ran.
    assert [r["cellId"] for r in st["results"]] == ["good", "bad"]


@pytest.mark.asyncio
async def test_get_run_status_unknown_job():
    out = json.loads(await colab_mcp.get_run_status.fn("nope"))
    assert "error" in out


@pytest.mark.asyncio
async def test_index_after_resolves_id(monkeypatch):
    async def fake_forward(tool_name, arguments):
        return json.dumps({"cells": [{"id": "a"}, {"id": "b"}, {"id": "c"}]})

    monkeypatch.setattr(colab_mcp, "_forward_or_stub", fake_forward)
    # Insert-after "a" -> index 1; after "c" -> index 3 (append).
    assert await colab_mcp._index_after("a") == 1
    assert await colab_mcp._index_after("c") == 3
    assert await colab_mcp._index_after("missing") is None


@pytest.mark.asyncio
async def test_add_code_cell_after_id(monkeypatch):
    monkeypatch.setattr(colab_mcp, "_proxy_client", _connected_proxy())
    sent = {}

    async def fake_forward(tool_name, arguments):
        if tool_name == "get_cells":
            return json.dumps({"cells": [{"id": "a"}, {"id": "b"}]})
        sent[tool_name] = arguments
        return "{}"

    monkeypatch.setattr(colab_mcp, "_forward_or_stub", fake_forward)
    await colab_mcp.add_code_cell.fn(code="x", afterCellId="a")
    assert sent["add_code_cell"]["cellIndex"] == 1  # resolved from id "a"

    bad = await colab_mcp.add_code_cell.fn(code="x", afterCellId="nope")
    assert "No such cellId" in bad

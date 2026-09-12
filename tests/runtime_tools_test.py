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

import json
from unittest.mock import Mock

import pytest

import colab_mcp
from colab_mcp.client import Accelerator, Variant


class _FakeAssignment:
    def __init__(self, endpoint, accelerator):
        self.endpoint = endpoint
        self.accelerator = accelerator


@pytest.fixture
def mock_client(monkeypatch):
    client = Mock()
    monkeypatch.setattr(colab_mcp, "_colab_client", client)
    return client


# --- not-initialized guard (shared by all three) ---


@pytest.mark.asyncio
async def test_runtime_tools_require_oauth(monkeypatch):
    monkeypatch.setattr(colab_mcp, "_colab_client", None)
    for out in (
        await colab_mcp.runtime_change.fn(),
        await colab_mcp.runtime_status.fn(),
        await colab_mcp.runtime_stop.fn(),
    ):
        assert "not initialized" in out


# --- runtime_change ---


@pytest.mark.asyncio
async def test_runtime_change_gpu(mock_client):
    mock_client.list_assignments.return_value = []
    mock_client.assign.return_value = _FakeAssignment("gpu-t4-xyz", Accelerator.T4)
    out = await colab_mcp.runtime_change.fn("T4")
    assert "gpu-t4-xyz" in out
    # GPU variant selected for T4.
    assert mock_client.assign.call_args[0][1] == Variant.GPU


@pytest.mark.asyncio
async def test_runtime_change_tpu_variant(mock_client):
    mock_client.list_assignments.return_value = []
    mock_client.assign.return_value = _FakeAssignment("tpu-xyz", Accelerator.V28)
    await colab_mcp.runtime_change.fn("V2-8")
    # V-prefixed accelerators must map to the TPU variant.
    assert mock_client.assign.call_args[0][1] == Variant.TPU


@pytest.mark.asyncio
async def test_runtime_change_none_variant(mock_client):
    mock_client.list_assignments.return_value = []
    mock_client.assign.return_value = _FakeAssignment("cpu-xyz", Accelerator.NONE)
    await colab_mcp.runtime_change.fn("NONE")
    assert mock_client.assign.call_args[0][1] == Variant.DEFAULT


@pytest.mark.asyncio
async def test_runtime_change_unwraps_dict_result(mock_client):
    mock_client.list_assignments.return_value = []
    # assign() can return {"assignment": Assignment} when a VM already exists.
    mock_client.assign.return_value = {
        "assignment": _FakeAssignment("existing-ep", Accelerator.T4),
        "is_new": False,
    }
    out = await colab_mcp.runtime_change.fn("T4")
    assert "existing-ep" in out


@pytest.mark.asyncio
async def test_runtime_change_bad_accelerator(mock_client):
    out = await colab_mcp.runtime_change.fn("BOGUS")
    assert "Failed to change runtime" in out


# --- runtime_status ---


@pytest.mark.asyncio
async def test_runtime_status_aggregates(mock_client):
    mock_client.get_subscription_tier.return_value = Mock(name="PRO")
    mock_client.get_subscription_tier.return_value.name = "PRO"
    ccu = Mock()
    ccu.current_balance = 42.5
    ccu.consumption_rate_hourly = 1.8
    ccu.assignments_count = 1
    mock_client.get_ccu_info.return_value = ccu
    mock_client.list_assignments.return_value = [
        _FakeAssignment("gpu-a100-1", Accelerator.A100)
    ]
    out = json.loads(await colab_mcp.runtime_status.fn())
    assert out["subscription_tier"] == "PRO"
    assert out["credit_balance"] == 42.5
    assert out["consumption_rate_hourly"] == 1.8
    assert out["assignments"] == [{"accelerator": "A100", "endpoint": "gpu-a100-1"}]


@pytest.mark.asyncio
async def test_runtime_status_partial_error(mock_client):
    # A failing sub-call is captured inline, not fatal to the whole report.
    mock_client.get_subscription_tier.side_effect = RuntimeError("boom")
    mock_client.get_ccu_info.return_value = Mock(
        current_balance=1.0, consumption_rate_hourly=0.0, assignments_count=0
    )
    mock_client.list_assignments.return_value = []
    out = json.loads(await colab_mcp.runtime_status.fn())
    assert "error: boom" in out["subscription_tier"]
    assert out["assignments"] == []


# --- runtime_stop ---


@pytest.mark.asyncio
async def test_runtime_stop_unassigns_all(mock_client):
    mock_client.list_assignments.return_value = [
        _FakeAssignment("ep1", Accelerator.T4),
        _FakeAssignment("ep2", Accelerator.L4),
    ]
    out = await colab_mcp.runtime_stop.fn()
    assert "Stopped 2 runtime(s)" in out
    assert mock_client.unassign.call_count == 2


@pytest.mark.asyncio
async def test_runtime_stop_none_assigned(mock_client):
    mock_client.list_assignments.return_value = []
    out = await colab_mcp.runtime_stop.fn()
    assert "No runtimes currently assigned" in out
    mock_client.unassign.assert_not_called()

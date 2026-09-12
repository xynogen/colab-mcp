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

from urllib.parse import parse_qs, urlsplit

from colab_mcp import _build_colab_url


class _FakeWSS:
    port = 8085
    token = "TKN"


def test_empty_url_creates_new_notebook():
    url = _build_colab_url("", _FakeWSS())
    parts = urlsplit(url)
    # #create=true -> fresh untitled notebook, not the shared scratch nb.
    assert parts.path == "/"
    assert "create=true" in parts.fragment
    assert parse_qs(parts.query)["p"] == ["8085"]
    assert "mcpProxyToken=TKN" in parts.fragment
    assert "mcpProxyPort=8085" in parts.fragment


def test_blank_whitespace_treated_as_empty():
    assert "create=true" in _build_colab_url("   ", _FakeWSS())


def test_drive_url_preserves_path_and_fragment():
    src = "https://colab.research.google.com/drive/ABC#scrollTo=x"
    parts = urlsplit(_build_colab_url(src, _FakeWSS()))
    assert parts.path == "/drive/ABC"
    # User fragment kept, proxy token/port appended after it.
    assert parts.fragment == "scrollTo=x&mcpProxyToken=TKN&mcpProxyPort=8085"
    assert parse_qs(parts.query)["p"] == ["8085"]


def test_existing_query_is_preserved():
    src = "https://colab.research.google.com/drive/ABC?authuser=1#scrollTo=x"
    q = parse_qs(urlsplit(_build_colab_url(src, _FakeWSS())).query)
    assert q["authuser"] == ["1"]
    assert q["p"] == ["8085"]

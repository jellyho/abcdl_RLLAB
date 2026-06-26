"""Tests for abcdl.hf — network-free (monkeypatches _hub)."""

from __future__ import annotations

import types

import abcdl.hf as hf


def test_push_uses_branch_and_tag(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(hf, "_hub", lambda: types.SimpleNamespace(
        create_repo=lambda **k: calls.setdefault("create_repo", k),
        create_branch=lambda **k: calls.setdefault("create_branch", k),
        upload_folder=lambda **k: calls.setdefault("upload_folder", k),
        create_tag=lambda **k: calls.setdefault("create_tag", k),
    ))
    (tmp_path / "f.txt").write_text("x")
    hf.push("jellyho/yam", str(tmp_path), fmt="abcdl", version="v1")
    assert calls["create_branch"]["branch"] == "abcdl"
    assert calls["upload_folder"]["revision"] == "abcdl"
    assert calls["create_tag"]["tag"] == "v1"
    assert calls["create_tag"]["revision"] == "abcdl"
    assert calls["upload_folder"]["repo_type"] == "dataset"


def test_pull_prefers_version_tag(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(hf, "_hub", lambda: types.SimpleNamespace(
        snapshot_download=lambda **k: seen.update(k) or str(tmp_path)))
    out = hf.pull("jellyho/yam", fmt="abcdl", version="v2", dest=str(tmp_path))
    assert seen["revision"] == "v2"
    assert seen["repo_type"] == "dataset"
    assert out == str(tmp_path)

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
    hf.push("jellyho/yam", str(tmp_path), fmt="abcdl_224", version="v1")
    assert calls["create_branch"]["branch"] == "abcdl_224"
    assert calls["upload_folder"]["revision"] == "abcdl_224"
    # tag is format-scoped so versions never collide across format branches
    assert calls["create_tag"]["tag"] == "abcdl_224-v1"
    assert calls["create_tag"]["revision"] == "abcdl_224"
    assert calls["create_tag"]["exist_ok"] is True
    assert calls["upload_folder"]["repo_type"] == "dataset"


def test_pull_latest_uses_branch_and_version_uses_scoped_tag(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(hf, "_hub", lambda: types.SimpleNamespace(
        snapshot_download=lambda **k: seen.update(k) or str(tmp_path)))
    hf.pull("jellyho/yam", fmt="mcap", version="latest", dest=str(tmp_path))
    assert seen["revision"] == "mcap"  # latest -> branch head
    out = hf.pull("jellyho/yam", fmt="abcdl_224", version="v2", dest=str(tmp_path))
    assert seen["revision"] == "abcdl_224-v2"  # versioned -> format-scoped tag
    assert seen["repo_type"] == "dataset"
    assert out == str(tmp_path)


def test_list_versions_filters_and_strips_format(monkeypatch):
    Ref = types.SimpleNamespace
    refs = types.SimpleNamespace(
        branches=[Ref(name="main"), Ref(name="mcap"), Ref(name="abcdl_224")],
        tags=[Ref(name="abcdl_224-v1"), Ref(name="abcdl_224-v2"), Ref(name="mcap-v1")],
    )
    monkeypatch.setattr(hf, "_hub", lambda: types.SimpleNamespace(
        list_repo_refs=lambda *a, **k: refs))
    allv = hf.list_versions("jellyho/yam")
    assert set(allv["branches"]) == {"main", "mcap", "abcdl_224"}
    scoped = hf.list_versions("jellyho/yam", fmt="abcdl_224")
    assert scoped["branches"] == ["abcdl_224"]
    assert sorted(scoped["tags"]) == ["v1", "v2"]  # prefix stripped, mcap-v1 excluded

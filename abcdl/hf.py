"""Publish/fetch abcdl & mcap datasets on the HuggingFace Hub (format=branch, version=tag)."""

from __future__ import annotations

from typing import Optional


def _hub():
    try:
        import huggingface_hub as h
    except ImportError as e:
        raise ImportError("HF integration needs `pip install huggingface_hub`") from e
    return h


def push(
    repo_id: str,
    local_dir: str,
    fmt: str,
    version: str,
    message: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Upload *local_dir* to *repo_id* on the Hub.

    The dataset format (e.g. ``"abcdl"``) is stored as a branch; the version
    (e.g. ``"v1"``) is stored as a tag pointing to that branch's HEAD.
    Returns the branch name (``fmt``).
    """
    h = _hub()
    h.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
    h.create_branch(
        repo_id=repo_id, repo_type="dataset", branch=fmt, exist_ok=True, token=token
    )
    h.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=local_dir,
        revision=fmt,
        commit_message=message or f"upload {fmt} {version}",
        token=token,
    )
    h.create_tag(
        repo_id=repo_id, repo_type="dataset", tag=version, revision=fmt, token=token
    )
    return fmt


def pull(
    repo_id: str,
    fmt: str,
    version: str = "latest",
    dest: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Download *repo_id* from the Hub and return the local directory path.

    When *version* is ``"latest"``, checks out the branch head (``fmt``);
    otherwise checks out the tag *version* directly.
    """
    h = _hub()
    revision = fmt if version == "latest" else version
    return h.snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=dest,
        token=token,
    )


def list_versions(
    repo_id: str, fmt: Optional[str] = None, token: Optional[str] = None
) -> dict:
    """Return ``{"branches": [...], "tags": [...]}`` for *repo_id*.

    If *fmt* is given, the branches list is filtered to only that branch name.
    """
    h = _hub()
    refs = h.list_repo_refs(repo_id, repo_type="dataset", token=token)
    branches = [b.name for b in refs.branches]
    tags = [t.name for t in refs.tags]
    if fmt is not None:
        branches = [b for b in branches if b == fmt]
    return {"branches": branches, "tags": tags}

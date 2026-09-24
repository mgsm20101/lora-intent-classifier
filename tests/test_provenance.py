"""Tests mock subprocess.run so no real git process is spawned and the
refusal logic can be exercised for clean, dirty, and non-git states."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.provenance import (
    DirtyWorktreeError,
    build_provenance,
    get_repo_state,
    require_clean_repo,
)


def _fake_run(rev_parse_ok: bool, porcelain_output: str):
    def run(args, cwd, capture_output, text, check):
        if args[1] == "rev-parse":
            returncode = 0 if rev_parse_ok else 128
            stdout = "abc123def456\n" if rev_parse_ok else ""
            return _Completed(returncode, stdout)
        if args[1] == "status":
            return _Completed(0, porcelain_output)
        raise AssertionError(f"unexpected git args: {args}")

    return run


class _Completed:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_get_repo_state_reports_clean_repo() -> None:
    with patch("subprocess.run", side_effect=_fake_run(rev_parse_ok=True, porcelain_output="")):
        state = get_repo_state(Path("."))

    assert state.in_git_repo is True
    assert state.worktree_clean is True
    assert state.source_commit_sha == "abc123def456"


def test_get_repo_state_reports_dirty_repo() -> None:
    with patch(
        "subprocess.run",
        side_effect=_fake_run(rev_parse_ok=True, porcelain_output=" M src/train_cpu.py\n"),
    ):
        state = get_repo_state(Path("."))

    assert state.in_git_repo is True
    assert state.worktree_clean is False


def test_get_repo_state_reports_not_a_repo() -> None:
    with patch("subprocess.run", side_effect=_fake_run(rev_parse_ok=False, porcelain_output="")):
        state = get_repo_state(Path("."))

    assert state.in_git_repo is False
    assert state.source_commit_sha is None


def test_require_clean_repo_raises_on_dirty_tree() -> None:
    with patch(
        "subprocess.run",
        side_effect=_fake_run(rev_parse_ok=True, porcelain_output=" M foo.py\n"),
    ):
        with pytest.raises(DirtyWorktreeError):
            require_clean_repo(Path("."))


def test_require_clean_repo_raises_outside_git() -> None:
    with patch("subprocess.run", side_effect=_fake_run(rev_parse_ok=False, porcelain_output="")):
        with pytest.raises(DirtyWorktreeError):
            require_clean_repo(Path("."))


def test_require_clean_repo_allows_dirty_tree_with_override() -> None:
    with patch(
        "subprocess.run",
        side_effect=_fake_run(rev_parse_ok=True, porcelain_output=" M foo.py\n"),
    ):
        state = require_clean_repo(Path("."), allow_dirty=True)

    assert state.worktree_clean is False


def test_require_clean_repo_passes_on_clean_tree() -> None:
    with patch("subprocess.run", side_effect=_fake_run(rev_parse_ok=True, porcelain_output="")):
        state = require_clean_repo(Path("."))

    assert state.worktree_clean is True
    assert state.source_commit_sha == "abc123def456"


def test_build_provenance_includes_expected_fields() -> None:
    with patch("subprocess.run", side_effect=_fake_run(rev_parse_ok=True, porcelain_output="")):
        provenance = build_provenance(Path("."), seed=20260923)

    assert provenance["seed"] == 20260923
    assert provenance["source_commit_sha"] == "abc123def456"
    assert provenance["worktree_clean"] is True
    assert provenance["hardware"] == "Windows 11, 15.9 GB RAM, CPU training (torch +cpu)"
    assert "torch" in provenance["library_versions"]
    assert "timestamp" in provenance

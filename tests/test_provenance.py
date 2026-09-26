"""Tests mock subprocess.run so no real git process is spawned and the
refusal logic can be exercised for clean, dirty, and non-git states."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.provenance import DirtyWorktreeError, build_provenance, git_state


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


CLEAN = _fake_run(rev_parse_ok=True, porcelain_output="")
DIRTY = _fake_run(rev_parse_ok=True, porcelain_output=" M src/train_cpu.py\n")
NOT_A_REPO = _fake_run(rev_parse_ok=False, porcelain_output="")


def test_git_state_reports_clean_repo() -> None:
    with patch("subprocess.run", side_effect=CLEAN):
        assert git_state(Path(".")) == ("abc123def456", True)


def test_git_state_reports_dirty_repo() -> None:
    with patch("subprocess.run", side_effect=DIRTY):
        assert git_state(Path(".")) == ("abc123def456", False)


def test_git_state_reports_not_a_repo() -> None:
    with patch("subprocess.run", side_effect=NOT_A_REPO):
        assert git_state(Path(".")) == (None, False)


def test_build_provenance_raises_on_dirty_tree() -> None:
    with patch("subprocess.run", side_effect=DIRTY):
        with pytest.raises(DirtyWorktreeError, match="uncommitted changes"):
            build_provenance(Path("."), seed=1)


def test_build_provenance_raises_outside_git() -> None:
    with patch("subprocess.run", side_effect=NOT_A_REPO):
        with pytest.raises(DirtyWorktreeError, match="not inside a git repository"):
            build_provenance(Path("."), seed=1)


def test_build_provenance_allows_dirty_tree_with_override() -> None:
    with patch("subprocess.run", side_effect=DIRTY):
        provenance = build_provenance(Path("."), seed=1, allow_dirty=True)

    assert provenance["worktree_clean"] is False


def test_build_provenance_on_clean_tree_includes_expected_fields() -> None:
    with patch("subprocess.run", side_effect=CLEAN):
        provenance = build_provenance(Path("."), seed=20260923)

    assert provenance["seed"] == 20260923
    assert provenance["source_commit_sha"] == "abc123def456"
    assert provenance["worktree_clean"] is True
    assert provenance["hardware"] == "Windows 11, 15.9 GB RAM, CPU training (torch +cpu)"
    assert "torch" in provenance["library_versions"]
    assert "timestamp" in provenance

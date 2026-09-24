"""Git provenance and the refusal-to-run-on-a-dirty-tree guard.

The measured run's results are only meaningful if they can be tied to an
exact, inspectable commit. This module resolves that commit and the
worktree's clean/dirty state, and raises before training starts if the
conditions are not met and the caller has not explicitly overridden them.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HARDWARE_STRING = "Windows 11, 15.9 GB RAM, CPU training (torch +cpu)"


class DirtyWorktreeError(RuntimeError):
    """Raised when the run would produce results with no fixed commit to cite."""


@dataclass(frozen=True)
class RepoState:
    source_commit_sha: str | None
    worktree_clean: bool
    in_git_repo: bool


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def get_repo_state(cwd: Path) -> RepoState:
    """Inspect `cwd` for a HEAD sha and a clean/dirty worktree.

    Returns `in_git_repo=False` (rather than raising) when `cwd` is not
    inside a git repository at all — the caller decides what to do with it.
    """
    head = _run_git(["rev-parse", "HEAD"], cwd)
    if head.returncode != 0:
        return RepoState(source_commit_sha=None, worktree_clean=False, in_git_repo=False)

    status = _run_git(["status", "--porcelain"], cwd)
    worktree_clean = status.returncode == 0 and status.stdout.strip() == ""
    return RepoState(
        source_commit_sha=head.stdout.strip(),
        worktree_clean=worktree_clean,
        in_git_repo=True,
    )


def require_clean_repo(cwd: Path, allow_dirty: bool = False) -> RepoState:
    """Enforce the "measured run needs a fixed, clean commit" rule.

    Raises `DirtyWorktreeError` when the tree is outside git or dirty and
    `allow_dirty` is False. `allow_dirty=True` bypasses the check entirely
    (used for local iteration, never for results meant to be cited).
    """
    state = get_repo_state(cwd)
    if allow_dirty:
        return state
    if not state.in_git_repo:
        raise DirtyWorktreeError(
            f"{cwd} is not inside a git repository; results would have no "
            "source_commit_sha to cite. Re-run with --allow-dirty to override."
        )
    if not state.worktree_clean:
        raise DirtyWorktreeError(
            "the worktree has uncommitted changes; results would not be "
            "reproducible from a fixed commit. Commit or stash first, or "
            "re-run with --allow-dirty to override."
        )
    return state


def library_versions() -> dict[str, str]:
    import peft
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


def build_provenance(cwd: Path, seed: int, allow_dirty: bool = False) -> dict:
    """Assemble the provenance block written alongside every measured run."""
    state = require_clean_repo(cwd, allow_dirty=allow_dirty)
    return {
        "seed": seed,
        "library_versions": library_versions(),
        "hardware": HARDWARE_STRING,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_commit_sha": state.source_commit_sha,
        "worktree_clean": state.worktree_clean,
    }

"""Git provenance and the refusal-to-run-on-a-dirty-tree guard.

The measured run's results are only meaningful if they can be tied to an
exact, inspectable commit. `build_provenance` resolves that commit and the
worktree's clean/dirty state, and raises before training starts if the
conditions are not met and the caller has not explicitly overridden them.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

HARDWARE_STRING = "Windows 11, 15.9 GB RAM, CPU training (torch +cpu)"


class DirtyWorktreeError(RuntimeError):
    """Raised when the run would produce results with no fixed commit to cite."""


def git_state(cwd: Path) -> tuple[str | None, bool]:
    """Return (HEAD sha, worktree clean) for `cwd`; sha is None outside git."""

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return None, False
    status = git("status", "--porcelain")
    return head.stdout.strip(), status.returncode == 0 and status.stdout.strip() == ""


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
    """Assemble the provenance block written alongside every measured run.

    Raises `DirtyWorktreeError` when `cwd` is outside git or dirty, unless
    `allow_dirty=True` (local iteration, never for results meant to be cited).
    """
    sha, clean = git_state(cwd)
    if not allow_dirty and sha is None:
        raise DirtyWorktreeError(
            f"{cwd} is not inside a git repository; results would have no "
            "source_commit_sha to cite. Re-run with --allow-dirty to override."
        )
    if not allow_dirty and not clean:
        raise DirtyWorktreeError(
            "the worktree has uncommitted changes; results would not be "
            "reproducible from a fixed commit. Commit or stash first, or "
            "re-run with --allow-dirty to override."
        )
    return {
        "seed": seed,
        "library_versions": library_versions(),
        "hardware": HARDWARE_STRING,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_commit_sha": sha,
        "worktree_clean": clean,
    }

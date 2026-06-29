"""Versioned checkpoint directories + resume resolution.

Each training run gets its own ``version_<N>`` subdirectory under
``<output_dir>/<experiment_name>/`` (aligned with the TensorBoard logger's
version), so re-running the same ``experiment_name`` never silently overwrites a
previous run's checkpoints / ``last.ckpt``.

Resume rules (``resolve_run``):
  * explicit ``load`` + ``resume=true``  -> full-state resume from that exact
    path; if the path lives in a ``version_<N>`` directory the run continues in
    that same directory (so preemption truly continues in place), otherwise a
    fresh version is started next to ``exp_ckpt_base``.
  * explicit ``load`` + ``resume=false`` -> weights-only load from that path into
    a brand-new version (fine-tuning never clobbers the source run).
  * ``auto_resume`` (no ``load``) -> resume the latest ``version_<N>/last.ckpt``
    found under ``exp_ckpt_base``, in place (SLURM requeue / preemption).
  * otherwise -> a fresh next version.

This module is pure stdlib (os/re) so it can be unit-tested without torch,
Lightning, or Hydra.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

_VERSION_RE = re.compile(r"version_(\d+)")
_VERSION_DIR_RE = re.compile(r"^version_(\d+)$")


def list_versions(base_dir: Optional[str]) -> list[int]:
    """Sorted integer versions of ``version_<N>`` subdirectories of ``base_dir``."""
    if not base_dir or not os.path.isdir(base_dir):
        return []
    out: list[int] = []
    for name in os.listdir(base_dir):
        m = _VERSION_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(base_dir, name)):
            out.append(int(m.group(1)))
    return sorted(out)


def parse_version_from_path(path) -> Optional[int]:
    """Return the version of the innermost ``version_<N>`` segment in ``path``."""
    if not path:
        return None
    matches = _VERSION_RE.findall(str(path))
    return int(matches[-1]) if matches else None


def next_version(*base_dirs: Optional[str]) -> int:
    """Smallest unused version across all given bases (max existing + 1, else 0)."""
    versions: list[int] = []
    for b in base_dirs:
        versions.extend(list_versions(b))
    return (max(versions) + 1) if versions else 0


def latest_last_ckpt(exp_ckpt_base: str) -> Optional[tuple[int, str]]:
    """Highest ``version_<N>`` under ``exp_ckpt_base`` that holds a ``last.ckpt``."""
    best: Optional[tuple[int, str]] = None
    for v in list_versions(exp_ckpt_base):
        p = os.path.join(exp_ckpt_base, f"version_{v}", "last.ckpt")
        if os.path.isfile(p):
            best = (v, p)
    return best


@dataclass
class RunResolution:
    version: int
    resume_ckpt: Optional[str]   # full-state resume -> trainer.fit(ckpt_path=...)
    weights_only: Optional[str]  # weights-only load -> fresh optimizer/schedule
    ckpt_dir: str                # where ModelCheckpoint writes for this run


def _version_dir(exp_ckpt_base: str, version: int) -> str:
    return os.path.join(exp_ckpt_base, f"version_{version}")


def resolve_run(
    *,
    load: Optional[str],
    resume: bool,
    auto_resume: bool,
    exp_ckpt_base: str,
    log_exp_base: Optional[str] = None,
) -> RunResolution:
    """Resolve the version + resume target for a training run. See module docstring."""
    load = load or None
    resume = bool(resume)
    auto_resume = bool(auto_resume)

    if load:
        if resume:
            v = parse_version_from_path(load)
            if v is not None:
                # Continue the existing run in the checkpoint's own directory.
                ckpt_dir = os.path.dirname(os.path.abspath(str(load)))
            else:
                v = next_version(exp_ckpt_base, log_exp_base)
                ckpt_dir = _version_dir(exp_ckpt_base, v)
            return RunResolution(v, str(load), None, ckpt_dir)
        # Weights-only fine-tune: start a brand-new versioned run.
        v = next_version(exp_ckpt_base, log_exp_base)
        return RunResolution(v, None, str(load), _version_dir(exp_ckpt_base, v))

    if auto_resume:
        found = latest_last_ckpt(exp_ckpt_base)
        if found is not None:
            v, last = found
            return RunResolution(v, last, None, _version_dir(exp_ckpt_base, v))

    v = next_version(exp_ckpt_base, log_exp_base)
    return RunResolution(v, None, None, _version_dir(exp_ckpt_base, v))

"""Tests for the coarse-to-fine ``_stage_schedule`` (stage + ramp ``mix``).

Imports the LightningModule, so it needs pytorch_lightning (and the loss deps it
imports). Built with ``eval_mode=True`` and a dummy model so no VGG/LPIPS weights
are constructed or downloaded.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")
# model_wrapper imports the loss module at import time (lpips/scipy/torchvision).
pytest.importorskip("lpips")

import torch.nn as nn

from globalsplat.model.model_wrapper import GlobalSplatModule


def _module(boundaries, ramp, final_stage):
    return GlobalSplatModule(
        model=nn.Identity(),
        eval_mode=True,                 # skips building the perceptual loss
        stage_boundaries=boundaries,
        stage_ramp_iters=ramp,
        final_stage=final_stage,
    )


def test_stage_progression_and_mix():
    m = _module(boundaries=(10, 20, 50), ramp=4, final_stage=3)
    sched = m._stage_schedule

    # before the first boundary: stage 0, fully mixed in
    assert sched(0) == (0, 1.0)
    assert sched(9) == (0, 1.0)

    # at a boundary the new stage starts ramping from mix=0
    assert sched(10) == (1, 0.0)
    assert sched(12) == (1, 0.5)      # t=2 of ramp 4
    assert sched(14) == (1, 1.0)      # ramp complete

    assert sched(20)[0] == 2 and sched(20)[1] == 0.0
    assert sched(50)[0] == 3 and sched(50)[1] == 0.0

    # well past the last boundary: top stage, fully mixed
    assert sched(1000) == (3, 1.0)


def test_final_stage_clips_boundaries():
    # final_stage=1 keeps only the first boundary, so stage never exceeds 1.
    m = _module(boundaries=(10, 20, 50), ramp=0, final_stage=1)
    assert m.stage_boundaries == (10,)
    assert m._stage_schedule(1000)[0] == 1


if __name__ == "__main__":
    test_stage_progression_and_mix()
    test_final_stage_clips_boundaries()
    print("test_stage_schedule: OK")

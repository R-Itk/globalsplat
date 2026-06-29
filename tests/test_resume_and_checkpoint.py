"""Tests for the two release fixes:

  1. Saved checkpoints contain only the model's used weights (the frozen
     perceptual/LPIPS loss nets under ``render_criterion.*`` are stripped).
  2. Resume continues the data schedule: the upstream ``StepTracker`` step is
     checkpointed/restored, and the train shuffle seed advances with the epoch
     and the resumed step instead of replaying from the start.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")
pytest.importorskip("lpips")

import torch
import torch.nn as nn

from globalsplat.model.model_wrapper import GlobalSplatModule
from globalsplat.dataset.data_module import UpstreamBackedDataModule


# --- 1. checkpoint slimming -------------------------------------------------

def test_on_save_strips_loss_net_unconditionally():
    m = GlobalSplatModule(model=nn.Identity(), eval_mode=True)
    # Attach a loss net with a *trainable* param to prove the strip no longer
    # depends on a frozen-check (the old guard skipped dropping in that case).
    m.render_criterion = nn.Linear(2, 2)  # params default requires_grad=True

    ckpt = {
        "state_dict": {
            "model.scene_tokens": torch.zeros(1),
            "model.gaussian_decoder.patch_center_bias": torch.zeros(1, 1, 3),
            "render_criterion.weight": torch.zeros(2, 2),
            "render_criterion.bias": torch.zeros(2),
        }
    }
    m.on_save_checkpoint(ckpt)
    keys = set(ckpt["state_dict"])
    assert keys == {"model.scene_tokens", "model.gaussian_decoder.patch_center_bias"}
    assert not any(k.startswith("render_criterion.") for k in keys)


def test_on_load_reinjects_loss_net_for_strict_resume():
    m = GlobalSplatModule(model=nn.Identity(), eval_mode=True)
    m.render_criterion = nn.Linear(2, 2)
    # A slimmed checkpoint (loss net dropped) must round-trip to a strict load:
    # on_load re-injects the missing render_criterion.* keys from the live module.
    ckpt = {"state_dict": {"model.scene_tokens": torch.zeros(1)}}
    m.on_load_checkpoint(ckpt)
    assert any(k.startswith("render_criterion.") for k in ckpt["state_dict"])


# --- 2. resume: step tracker + seed ----------------------------------------

class _FakeTracker:
    def __init__(self, step=0):
        self._step = int(step)

    def get_step(self):
        return self._step

    def set_step(self, step):
        self._step = int(step)


def _dm(tracker):
    # upstream_dm is only touched inside _make_loader, so None is fine here.
    return UpstreamBackedDataModule(upstream_dm=None, step_tracker=tracker)


def test_step_tracker_is_checkpointed():
    dm = _dm(_FakeTracker(step=12345))
    assert dm.state_dict() == {"step_tracker_step": 12345}


def test_load_state_dict_restores_step_and_resume_marker():
    tracker = _FakeTracker(step=0)
    dm = _dm(tracker)
    dm.load_state_dict({"step_tracker_step": 5000})
    assert dm._resumed_step == 5000
    # applied to the live tracker immediately (before the first resumed batch)
    assert tracker.get_step() == 5000


def test_missing_state_is_graceful():
    tracker = _FakeTracker(step=0)
    dm = _dm(tracker)
    dm.load_state_dict({})  # old checkpoint without the key
    assert dm._resumed_step == 0
    assert tracker.get_step() == 0


def test_train_seed_advances_and_is_deterministic():
    seed = UpstreamBackedDataModule._train_seed
    base, rank = 42, 0

    # deterministic for the same (epoch, resumed_step)
    assert seed(base, rank, 3, 100) == seed(base, rank, 3, 100)
    # epoch changes the stream
    assert seed(base, rank, 0, 0) != seed(base, rank, 1, 0)
    # resuming mid-stream advances to a different stream than the fresh epoch
    assert seed(base, rank, 2, 0) != seed(base, rank, 2, 777)
    # rank separates DDP replicas
    assert seed(base, 0, 0, 0) != seed(base, 1, 0, 0)


if __name__ == "__main__":
    test_on_save_strips_loss_net_unconditionally()
    test_on_load_reinjects_loss_net_for_strict_resume()
    test_step_tracker_is_checkpointed()
    test_load_state_dict_restores_step_and_resume_marker()
    test_missing_state_is_graceful()
    test_train_seed_advances_and_is_deterministic()
    print("test_resume_and_checkpoint: OK")

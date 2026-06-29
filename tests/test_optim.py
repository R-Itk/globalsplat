"""Tests for the optimizer/param-group policy in ``globalsplat.model.optim``.

Verifies that weight decay is applied only to >=2-D weight matrices and that
norm affine params, biases, embedding-like banks, and frozen params are handled
as documented. Needs only torch.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from globalsplat.model.optim import build_param_groups, build_optimizer_and_scheduler


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)            # weight (decay) + bias (no decay)
        self.ln = nn.LayerNorm(4)             # weight + bias (both no decay)
        self.scene_tokens = nn.Parameter(torch.randn(3, 4))  # bank (no decay)
        self.frozen = nn.Parameter(torch.randn(4, 4), requires_grad=False)


def _ids(params):
    return {id(p) for p in params}


def test_param_groups_split():
    net = _Net()
    groups = build_param_groups(net, weight_decay=0.1)

    assert len(groups) == 2
    decay = next(g for g in groups if g["weight_decay"] == 0.1)
    no_decay = next(g for g in groups if g["weight_decay"] == 0.0)

    decay_ids = _ids(decay["params"])
    no_decay_ids = _ids(no_decay["params"])

    # Only the 2-D Linear weight is decayed.
    assert decay_ids == {id(net.lin.weight)}

    # Norm weight+bias, Linear bias, and the named bank are excluded from decay.
    for p in (net.ln.weight, net.ln.bias, net.lin.bias, net.scene_tokens):
        assert id(p) in no_decay_ids

    # Frozen params never enter any group.
    all_ids = decay_ids | no_decay_ids
    assert id(net.frozen) not in all_ids


def test_decay_biases_flag():
    net = _Net()
    groups = build_param_groups(net, weight_decay=0.1, decay_biases=True)
    decay = next(g for g in groups if g["weight_decay"] == 0.1)
    decay_ids = _ids(decay["params"])
    # Non-norm bias now decays; norm bias still excluded.
    assert id(net.lin.bias) in decay_ids
    assert id(net.ln.bias) not in decay_ids


def test_scheduler_interval_step_vs_epoch():
    net = _Net()

    step_cfg = build_optimizer_and_scheduler(
        net, lr=1e-3, weight_decay=0.0, total_steps=1000, fused=False
    )
    assert "optimizer" in step_cfg
    assert step_cfg["lr_scheduler"]["interval"] == "step"

    epoch_cfg = build_optimizer_and_scheduler(
        net, lr=1e-3, weight_decay=0.0, total_steps=None, max_epochs=10, fused=False
    )
    assert epoch_cfg["lr_scheduler"]["interval"] == "epoch"


if __name__ == "__main__":
    test_param_groups_split()
    test_decay_biases_flag()
    test_scheduler_interval_step_vs_epoch()
    print("test_optim: OK")

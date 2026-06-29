"""Optimizer + LR-schedule construction for GlobalSplat.

Split out of ``model_wrapper.py`` so the parameter-group policy (what gets
weight decay) and the warmup -> cosine schedule live in one place and can be
unit-tested without instantiating a LightningModule.

The previous ``configure_optimizers`` applied weight decay to *every* parameter
(LayerNorm/LayerScale gains, biases, and the learnable token/embedding banks).
``build_param_groups`` fixes that: decay is applied only to the >=2-D weight
matrices; norms, biases, and the embedding-like banks are excluded.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


# Normalization layers whose affine params (weight + bias) must never be decayed.
# Detected by module *type* rather than guessed from the parameter name, since a
# ``.bias`` also exists on regular Linear/Conv layers.
_NORM_TYPES: tuple = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)
if hasattr(nn, "RMSNorm"):  # torch >= 2.4
    _NORM_TYPES = _NORM_TYPES + (nn.RMSNorm,)

# Leaf names of >=2-D, embedding-like banks that must not be decayed either:
# decaying a learnable token/embedding bank just pulls it toward zero.
_NO_DECAY_PARAM_NAMES = frozenset(
    {
        "scene_tokens",                                          # scene-token bank
        "scale_emb", "aux_query_tokens", "slot_regs", "mem_regs",  # slot-encoder banks
    }
)


def _norm_param_ids(module: nn.Module) -> set:
    """``id()`` of every parameter owned directly by a normalization layer.

    Custom per-channel gains (e.g. ``LayerScale.gamma``) are matched by class
    name so we don't have to import every backbone's LayerScale variant.
    """
    ids = set()
    for m in module.modules():
        if isinstance(m, _NORM_TYPES) or m.__class__.__name__ == "LayerScale":
            ids.update(id(p) for p in m.parameters(recurse=False))
    return ids


def build_param_groups(
    module: nn.Module,
    weight_decay: float,
    *,
    no_decay_names: Iterable[str] = _NO_DECAY_PARAM_NAMES,
    decay_biases: bool = False,
) -> List[Dict[str, Any]]:
    """Split a module's *trainable* params into decay / no-decay groups.

    Weight decay is applied only to the >=2-D weight matrices (Linear/Conv
    weights). Excluded from decay:
      * params owned by a normalization layer (by module type, see
        ``_NORM_TYPES``) plus custom LayerScale gains -- weight *and* bias;
      * biases, by default. Set ``decay_biases=True`` to decay the non-norm ones
        (e.g. Linear biases); norm biases stay excluded either way;
      * the embedding-like banks named in ``no_decay_names``;
      * any remaining 1-D param (safety net for stray gains/scales).

    Frozen params (e.g. the VGG/LPIPS perceptual nets) are skipped entirely so
    they never enter the optimizer state.

    Returns a list suitable for ``torch.optim.AdamW(param_groups, lr=...)``.
    """
    no_decay_names = set(no_decay_names)
    norm_ids = _norm_param_ids(module)

    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []

    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        leaf = name.split(".")[-1]

        if id(p) in norm_ids or leaf in no_decay_names:
            no_decay.append(p)            # norm affine params + embedding banks
        elif leaf == "bias":
            (decay if decay_biases else no_decay).append(p)
        elif p.ndim <= 1:
            no_decay.append(p)            # stray 1-D gains / scales
        else:
            decay.append(p)               # the 2-D Linear/Conv weights

    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer_and_scheduler(
    module: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    total_steps: int | None = None,
    warmup_pct: float = 0.03,
    min_lr_ratio: float = 1.0 / 50.0,
    min_lr_floor: float = 1e-6,
    warmup_epochs_fallback: int = 5,
    max_epochs: int = 100,
    fused: bool | None = None,
) -> Dict[str, Any]:
    """Build AdamW + a warmup -> cosine schedule in Lightning's return format.

    If ``total_steps`` is known (``Trainer.estimated_stepping_batches``), the
    schedule advances per optimizer step; otherwise it falls back to a per-epoch
    schedule driven by ``max_epochs``.

    ``fused`` selects PyTorch's fused AdamW kernel. It is functionally the same
    AdamW update (only the kernel differs), so training results are unchanged up
    to floating-point ordering. Defaults to on when CUDA is available.
    """
    lr = float(lr)
    use_fused = (torch.cuda.is_available() if fused is None else bool(fused))
    try:
        opt = torch.optim.AdamW(build_param_groups(module, weight_decay), lr=lr, fused=use_fused)
    except (RuntimeError, ValueError):
        # Older torch or a non-CUDA/ineligible setup: fall back to the default impl.
        opt = torch.optim.AdamW(build_param_groups(module, weight_decay), lr=lr)
    eta_min = max(float(min_lr_floor), lr * float(min_lr_ratio))

    if total_steps and total_steps > 0:
        ws = max(1, int(float(warmup_pct) * total_steps))
        warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=ws)
        cosine = CosineAnnealingLR(opt, T_max=max(1, total_steps - ws), eta_min=eta_min)
        sched = SequentialLR(opt, [warmup, cosine], milestones=[ws])
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "name": "lr"},
        }

    we = int(warmup_epochs_fallback)
    warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=we)
    cosine = CosineAnnealingLR(opt, T_max=max(1, int(max_epochs) - we), eta_min=eta_min)
    sched = SequentialLR(opt, [warmup, cosine], milestones=[we])
    return {
        "optimizer": opt,
        "lr_scheduler": {"scheduler": sched, "interval": "epoch", "name": "lr"},
    }

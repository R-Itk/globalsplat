#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coarse-to-fine 3DGS token decoder with a learned gate.

This module predicts a fixed maximum number of Gaussian candidates per patch and
uses a gate, predicted from the geometry branch only, to reduce them in
block-wise groups according to the current stage.

The staged reduction is applied to:
    - positions
    - log-scales
    - rotations (6D)
    - spherical harmonics coefficients
    - opacity logits

Opacity is merged through weighted log-transmittance so that the reduction is
compatible with alpha compositing.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _clamp_int(x: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(x))))


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable computation of log(1 - exp(x)) for x <= 0.
    """
    threshold = -0.6931471805599453  # -log(2)
    return torch.where(x > threshold, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


# -----------------------------------------------------------------------------
# Regularizers
# -----------------------------------------------------------------------------

@torch.no_grad()
def margin_from_scale(
    gs_scale: torch.Tensor,
    *,
    mult: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Estimate a detached scalar margin from Gaussian scales.

    Args:
        gs_scale: [B, N, 3] Gaussian scales in world units.
        mult: Multiplicative factor applied to the median radius.
        eps: Minimum value for numerical stability.
    """
    radius = (gs_scale[..., 0] * gs_scale[..., 1] * gs_scale[..., 2]).clamp_min(eps).pow(1.0 / 3.0)
    return (float(mult) * radius.median()).detach()


def sh_softcap_reg(
    sh: torch.Tensor,
    cap: float = 5.0,
    tau: float = 1.0,
    p: float = 2.0,
) -> torch.Tensor:
    """
    Soft penalty for large spherical harmonics coefficients.

    Args:
        sh: [..., D] raw SH coefficients.
        cap: Target magnitude threshold.
        tau: Softness of the transition beyond the cap.
        p: Exponent of the penalty.
    """
    over = F.softplus((sh.abs() - float(cap)) / float(tau)) * float(tau)
    return over.pow(float(p)).mean()


# -----------------------------------------------------------------------------
# Gated block reduction helpers
# -----------------------------------------------------------------------------

def _block_softmax_weights(
    g_logits_full: torch.Tensor,
    G: int,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Convert per-child gate logits into block-local softmax weights.

    Args:
        g_logits_full: [B, P, M, 1]
        G: Number of groups kept at the current stage.
        tau: Softmax temperature.

    Returns:
        [B, P, G, bs, 1], where bs = M / G.
    """
    B, P, M, one = g_logits_full.shape
    assert one == 1
    assert M % G == 0

    block_size = M // G
    grouped_logits = g_logits_full.view(B, P, G, block_size, 1)
    return F.softmax(grouped_logits / float(tau), dim=3)


def _block_weighted_reduce(
    x_full: torch.Tensor,
    G: int,
    w: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted block reduction from M children to G groups.

    Args:
        x_full: [B, P, M, D]
        w: [B, P, G, bs, 1]

    Returns:
        [B, P, G, D]
    """
    B, P, M, D = x_full.shape
    block_size = M // G
    grouped = x_full.view(B, P, G, block_size, D)
    return (w * grouped).sum(dim=3)


def _block_alpha_union_logit_reduce_gated(
    op_logit_full: torch.Tensor,
    G: int,
    w: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Reduce opacity logits through weighted log-transmittance.

    The parent transmittance is defined as:
        log(1 - alpha_parent) = sum_i w_i * log(1 - alpha_i)

    Args:
        op_logit_full: [B, P, M, 1]
        G: Number of groups kept at the current stage.
        w: [B, P, G, bs, 1]

    Returns:
        Merged opacity logits with shape [B, P, G, 1].
    """
    B, P, M, one = op_logit_full.shape
    assert one == 1

    block_size = M // G
    op_grouped = op_logit_full.view(B, P, G, block_size, 1)

    log_u = F.logsigmoid(-op_grouped)
    log_u_parent = (w * log_u).sum(dim=3)

    log_u_parent = log_u_parent.clamp_max(-eps)
    return log1mexp(log_u_parent) - log_u_parent


# -----------------------------------------------------------------------------
# Main module
# -----------------------------------------------------------------------------

class TokenCoarseToFine3DGS(nn.Module):
    """
    Coarse-to-fine Gaussian decoder with gated staged reduction.

    The module predicts `M_max` Gaussian candidates per patch and reduces them to
    `G = 2^stage` active outputs using a learned gate derived only from the
    geometry branch. During stage transitions, outputs from the previous stage
    and current stage are blended using `mix`.

    Returned tensors are flattened over patches and active Gaussians.
    """

    def __init__(
        self,
        C: int,
        M_max: int = 16,
        op_bias0: float = -5.0,
        logscale_bias0: float = -2.0,
        patch_center_bias: Tuple[float, float, float] = (0.0, 0.0, 1.5),
        scale_min: float = 1e-4,
        scale_max: float = 0.05,
        renorm_rot6: bool = True,
        geo_readout_std: float = 1e-2,
        splat_readout_std: float = 1e-3,
        sh_degree: int = 0,
        sh_rest_scale: float = 0.25,
        gate_tau: float = 1.0,
        gate_entropy_w: float = 0.0,
    ):
        super().__init__()

        assert (M_max & (M_max - 1)) == 0, "M_max must be power of two."

        self.C = int(C)
        self.M_max = int(M_max)

        self.Smax = int(math.log2(self.M_max))
        self.stage = 0
        self.mix = 1.0

        self.op_bias0 = float(op_bias0)
        self.logscale_bias0 = float(logscale_bias0)

        self.max_scale = float(scale_max)
        self.log_scale_min = math.log(float(scale_min))
        self.log_scale_max = math.log(self.max_scale)

        self.renorm_rot6 = bool(renorm_rot6)

        self.register_buffer(
            "patch_center_bias",
            torch.tensor(patch_center_bias, dtype=torch.float32).view(1, 1, 3),
        )

        self.sh_degree = int(sh_degree)
        self.sh_dim = (self.sh_degree + 1) ** 2
        self.color_dim = 3 * self.sh_dim
        self.sh_rest_scale = float(sh_rest_scale)

        self.gate_tau = float(gate_tau)
        self.gate_entropy_w = float(gate_entropy_w)

        self.register_buffer(
            "static_rot6",
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32).view(1, 1, 1, 6),
            persistent=False,
        )

        # Geometry branch: position + log-scale / rotation / opacity.
        self.geo_pos_readout = nn.Linear(self.C, self.M_max * 3)
        self.geo_param_readout = nn.Linear(self.C, self.M_max * 10)

        # Appearance branch: spherical harmonics coefficients only.
        self.sh_readout = nn.Linear(self.C, self.M_max * self.color_dim)

        # Gate branch: one logit per child, predicted from geo tokens only.
        self.gate_readout = nn.Linear(self.C, self.M_max)

        with torch.no_grad():
            nn.init.trunc_normal_(self.geo_pos_readout.weight, std=float(geo_readout_std))
            nn.init.trunc_normal_(self.geo_param_readout.weight, std=float(geo_readout_std))
            nn.init.trunc_normal_(self.sh_readout.weight, std=float(splat_readout_std))
            nn.init.trunc_normal_(self.gate_readout.weight, std=float(geo_readout_std))

            for module in (self.geo_pos_readout, self.geo_param_readout, self.sh_readout, self.gate_readout):
                if module.bias is not None:
                    module.bias.zero_()

    @torch.no_grad()
    def set_stage(self, stage: int, mix: float = 1.0) -> None:
        self.stage = _clamp_int(int(stage), 0, self.Smax)
        self.mix = float(max(0.0, min(1.0, float(mix))))

    @staticmethod
    def _upsample_scale_volume_preserving(prev_log_s: torch.Tensor) -> torch.Tensor:
        """
        Split each parent scale into two children while preserving volume.
        """
        out = prev_log_s.repeat_interleave(2, dim=2)
        return out - (math.log(2.0) / 3.0)

    @staticmethod
    def _upsample_opacity_union_preserving(prev_logit: torch.Tensor) -> torch.Tensor:
        """
        Split each parent opacity into two children while preserving union alpha.
        """
        log_u = F.logsigmoid(-prev_logit)
        log_u_child = 0.5 * log_u
        child_logit = log1mexp(log_u_child) - log_u_child
        return child_logit.repeat_interleave(2, dim=2)

    def _reduce_with_mix_gated(
        self,
        x_full: torch.Tensor,
        stage: int,
        mix: float,
        w_curr: torch.Tensor,
        w_prev: Optional[torch.Tensor],
        *,
        upsample_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Reduce a full set of per-child predictions to the active stage and
        optionally blend it with the previous stage.
        """
        G = 1 << stage
        curr = _block_weighted_reduce(x_full, G, w_curr) if G != self.M_max else x_full

        if stage <= 0:
            return curr

        G_prev = G >> 1
        assert w_prev is not None

        prev = _block_weighted_reduce(x_full, G_prev, w_prev) if G_prev != self.M_max else x_full
        prev = prev.repeat_interleave(2, dim=2) if upsample_fn is None else upsample_fn(prev)

        return (1.0 - mix) * prev + mix * curr

    def forward(self, encoded_patch: Any):
        if isinstance(encoded_patch, (tuple, list)):
            splat_in, geo_in = encoded_patch
        else:
            splat_in = geo_in = encoded_patch

        B, P, C = geo_in.shape
        if C != self.C:
            raise ValueError(f"Expected C={self.C}, got {C}")

        # stage/mix are kept clamped by set_stage().
        stage = self.stage
        mix = self.mix
        G = 1 << stage
        G_prev = max(1, G >> 1)

        gate_logits_full = self.gate_readout(geo_in).view(B, P, self.M_max, 1)
        w_curr = _block_softmax_weights(gate_logits_full, G, tau=self.gate_tau)
        w_prev = _block_softmax_weights(gate_logits_full, G_prev, tau=self.gate_tau) if stage > 0 else None

        if self.gate_entropy_w != 0.0:
            probs = w_curr.clamp_min(1e-12)
            ent_curr = -(probs * probs.log()).sum(dim=3).mean()
            gate_ent = float(self.gate_entropy_w) * ent_curr
        else:
            gate_ent = gate_logits_full.sum() * 0.0

        # ------------------------------------------------------------------
        # Geometry branch
        # ------------------------------------------------------------------
        pos_full = self.geo_pos_readout(geo_in).view(B, P, self.M_max, 3)
        pos = self._reduce_with_mix_gated(pos_full, stage, mix, w_curr, w_prev)
        pos = pos + self.patch_center_bias.unsqueeze(2)

        param_full = self.geo_param_readout(geo_in).view(B, P, self.M_max, 10)
        log_s_full = param_full[..., 0:3] + self.logscale_bias0
        rot6_full = param_full[..., 3:9]
        op_full = param_full[..., 9:10] + self.op_bias0
        op_full = op_full - (1.0 - mix) * math.log(2.0)
        op_full = op_full.clamp(-13.0, 6.0)

        log_scale_too_large = F.relu(log_s_full - self.log_scale_max).pow(2).mean()
        log_s_full = log_s_full.clamp(self.log_scale_min - 1.0, self.log_scale_max + 2.0)

        def reduce_log_s_gated(x_full_: torch.Tensor, G_: int, w_: torch.Tensor) -> torch.Tensor:
            out = _block_weighted_reduce(x_full_, G_, w_) if G_ != self.M_max else x_full_
            block_size = self.M_max // G_
            if block_size > 1:
                out = out + (math.log(block_size) / 3.0)
            return out

        log_s_curr = reduce_log_s_gated(log_s_full, G, w_curr)
        if stage == 0:
            log_s = log_s_curr
        else:
            assert w_prev is not None
            log_s_prev = reduce_log_s_gated(log_s_full, G_prev, w_prev)
            log_s_prev = self._upsample_scale_volume_preserving(log_s_prev)
            log_s = (1.0 - mix) * log_s_prev + mix * log_s_curr

        op_curr = _block_alpha_union_logit_reduce_gated(op_full, G, w_curr)
        if stage == 0:
            op_logit = op_curr
        else:
            assert w_prev is not None
            op_prev = _block_alpha_union_logit_reduce_gated(op_full, G_prev, w_prev)
            op_prev = self._upsample_opacity_union_preserving(op_prev)
            op_logit = (1.0 - mix) * op_prev + mix * op_curr

        rot6 = self._reduce_with_mix_gated(rot6_full, stage, mix, w_curr, w_prev)

        # ------------------------------------------------------------------
        # Appearance branch
        # ------------------------------------------------------------------
        sh_full = self.sh_readout(splat_in).view(B, P, self.M_max, self.color_dim)
        sh_red = self._reduce_with_mix_gated(sh_full, stage, mix, w_curr, w_prev)

        # ------------------------------------------------------------------
        # Decode predictions
        # ------------------------------------------------------------------
        gs_scale = log_s.exp()

        rot6d = rot6 + self.static_rot6
        if self.renorm_rot6:
            r1 = F.normalize(rot6d[..., 0:3], dim=-1, eps=1e-8)
            r2 = F.normalize(rot6d[..., 3:6], dim=-1, eps=1e-8)
            rot6d = torch.cat([r1, r2], dim=-1)

        opacity = torch.sigmoid(op_logit).clamp(0.0, 0.99)
        sh = sh_red.view(B, P, G, self.sh_dim, 3)

        N = P * G
        pos = pos.reshape(B, N, 3)
        rot6d = rot6d.reshape(B, N, 6)
        gs_scale = gs_scale.reshape(B, N, 3)
        opacity = opacity.reshape(B, N, 1)
        sh = sh.reshape(B, N, self.sh_dim, 3)

        # ------------------------------------------------------------------
        # Regularization
        # ------------------------------------------------------------------
        gauge_rot = rot6_full.pow(2).mean()
        sh_cap_reg = sh_softcap_reg(sh_full)
        regs = 1e-2 * (log_scale_too_large + gauge_rot + sh_cap_reg) + gate_ent

        return pos, rot6d, gs_scale, sh, opacity, regs

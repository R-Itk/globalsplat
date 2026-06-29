"""Typed container for the model's output: a set of 3D Gaussians in the exact
layout the gsplat rasterizer (see ``model/rendering.py``) consumes.

This replaces the old ``SceneKeys`` string-enum dict. Fields are per-batch with
shape ``[B, N, ...]``; the camera dimension comes from the *target* cameras at
render time, not from here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class Gaussians:
    """A batch of 3D Gaussians ready for gsplat rasterization.

    Attributes:
        means:      ``[B, N, 3]`` world-space centers.
        scales:     ``[B, N, 3]`` per-axis scales (linear, already exponentiated).
        rotations:  ``[B, N, 6]`` 6D rotation; the renderer converts these to the
                    quaternions gsplat expects.
        sh:         ``[B, N, K, 3]`` spherical-harmonics coefficients, where
                    ``K = (sh_degree + 1) ** 2`` (gsplat's ``colors`` in SH mode).
        opacities:  ``[B, N, 1]`` opacities in ``[0, 1]``.
        reg:        scalar regularizer accumulated by the decoder (carried with the
                    Gaussians so the training loop can add it to the loss).
    """

    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    sh: torch.Tensor
    opacities: torch.Tensor
    reg: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.means.shape[0]

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[1]

    @property
    def sh_degree(self) -> int:
        """SH degree implied by the coefficient count K = (degree + 1)^2."""
        k = self.sh.shape[-2]
        return int(round(k ** 0.5)) - 1

    def __getitem__(self, idx) -> "Gaussians":
        """Index/slice along the batch dimension. ``reg`` is a scalar accumulated
        over the whole (sub)batch and is shared, not split."""
        return replace(
            self,
            means=self.means[idx],
            scales=self.scales[idx],
            rotations=self.rotations[idx],
            sh=self.sh[idx],
            opacities=self.opacities[idx],
        )

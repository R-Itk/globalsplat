"""Vendored transformer layers from VGGT (https://github.com/facebookresearch/vggt).

Only the three classes used by the GlobalSplat encoder are re-exported here.
These files are copied verbatim from VGGT's `vggt/layers/` and are licensed
under the Apache-2.0 / BSD terms in the upstream LICENSE (see NOTICE.md).
"""
from .block import Block
from .layer_scale import LayerScale
from .mlp import Mlp

__all__ = ["Block", "LayerScale", "Mlp"]

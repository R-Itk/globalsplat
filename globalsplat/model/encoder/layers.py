"""Re-export of the transformer building blocks used by the encoder.

The implementations live in the vendored ``backbone_layers`` package (copied
from VGGT). Importing them through this module keeps call sites short and gives
a single place to swap the backbone if needed.
"""
from .backbone_layers import Block, LayerScale, Mlp

__all__ = ["Block", "LayerScale", "Mlp"]

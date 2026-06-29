# Third-party code and licenses

This project vendors and depends on the following third-party code.

## Vendored (copied into this repository)

**VGGT** — `globalsplat/model/encoder/backbone_layers/` (`attention.py`, `block.py`,
`drop_path.py`, `layer_scale.py`, `mlp.py`).
Source: https://github.com/facebookresearch/vggt (commit `a288dd0`).
Copyright (c) Meta Platforms, Inc. and affiliates. Used under the upstream
Apache-2.0 / BSD terms; original license headers are retained in each file.

## Vendored upstream

**ZPressor / MVSplat / DepthSplat / pixelSplat** — `third_party/ZPressor/` (vendored, trimmed).
Source: https://github.com/ziplab/ZPressor (commit `81f3df8`).
Used for RE10K / DL3DV dataset loading and the bounded view sampler. See that
repository's LICENSE.

## Other components

- The VGG-19 perceptual loss (`globalsplat/loss/rendering_loss.py`) is adapted from the
  LVSM project and Long-LRM; attributions are in the file header.
- The 6D-rotation / quaternion helpers in `globalsplat/misc/geometry.py` are adapted
  from PyTorch3D (BSD); attribution is in the file.

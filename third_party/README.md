# Third-party dependencies

GlobalSplat reuses the **data loading, view sampling, and evaluation** code from
the ZPressor / MVSplat / DepthSplat family.

## ZPressor (vendored)

A trimmed copy of [ZPressor](https://github.com/ziplab/ZPressor) (commit
`81f3df8`) is vendored here under `third_party/ZPressor/` so the repo runs
out of the box. Large demo `assets/` were removed to keep it small; the code
(`mvsplat/`, `depthsplat/`, `pixelsplat/`, `zpressor/`) is intact.

The dataset configs point at it:

- `config/dataset/re10k.yaml`  -> `mvsplat_root: ./third_party/ZPressor/mvsplat`
- `config/dataset/dl3dv.yaml`  -> `depthsplat_root: ./third_party/ZPressor/depthsplat`

GlobalSplat imports the upstream modules as `src.*` (e.g.
`src.dataset.data_module.DataModule`, `src.evaluation.metrics`). This is why the
GlobalSplat package itself is named `globalsplat`, not `src` — so the two never
collide. Use **one** dataset (one upstream `src` root) per process.

To refresh to a newer upstream version, re-clone ZPressor over this directory.

## VGGT layers (vendored)

The three transformer layer classes used by the encoder (`Block`, `Mlp`,
`LayerScale`) are vendored under `globalsplat/model/encoder/backbone_layers/`
(VGGT commit `a288dd0`). See `../NOTICE.md`.

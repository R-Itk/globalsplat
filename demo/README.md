# Demo — runnable example scenes

Three RealEstate10K scenes bundled as **self-contained, runnable examples** — each
with its input views and camera parameters, plus a precomputed GlobalSplat-16K
reconstruction. You can view the result instantly in a browser, or re-run the
model on the bundled inputs yourself (no RE10K dataset needed).

These use the **no-opacity-loss 16K** checkpoint, whose Gaussians have solid
opacities that render cleanly in generic web viewers. (The 2K/16K-default/32K
checkpoints use much lower per-Gaussian opacity — they look correct in the gsplat
renderer but translucent in a free-orbit web viewer.)

## Layout

```
scenes/<name>/
  images/000.png …          input context views (256×256), spaced >=10 frames apart
  cameras.json              per-view pixel intrinsics + camera-to-world poses
  <name>.ply                precomputed GlobalSplat-16K output (16,384 Gaussians)
```

Bundled scenes (context views span the whole clip for visible parallax):
`5aca87f95a9412c6` (12 views), `322261824c4a3003` (9), `ffa95c3b40609c76` (12).

## 1. View in the browser (no install)

Open any web splat viewer and drag in a `scenes/<name>/<name>.ply` (or paste its
raw GitHub URL):

- **SuperSplat** — https://superspl.at/editor
- **antimatter15 / splat** — https://antimatter15.com/splat/
- **PlayCanvas Viewer** — https://playcanvas.com/viewer

These are standard INRIA-format 3DGS `.ply` (degree-3 SH, logit opacity, log
scale, `(w,x,y,z)` quaternion), so any Gaussian-Splatting viewer works.

## 2. Run the model yourself (no dataset)

The scene folders carry their own images + camera params, so you only need the
install and a checkpoint (see the [Pretrained models](../README.md#pretrained-models)
table):

Download the no-opacity 16K checkpoint (`globalsplat-re10k-16k-noopacity.ckpt`)
from the [Pretrained models](../README.md#pretrained-models) table, then:

```bash
python demo/run_demo.py \
    --scene demo/scenes/5aca87f95a9412c6 \
    --checkpoint globalsplat-re10k-16k-noopacity.ckpt
```

This loads the bundled views, runs the feed-forward model, and writes the `.ply`
next to the scene. Use `--experiment re10k_2k` / `re10k_32k` with the matching
checkpoint for the other variants.

## Regenerating the scenes (maintainers)

`export_demo.py` rebuilds the bundled scenes from the RE10K dataset (requires the
chunks). It picks context views evenly across each clip with a minimum frame gap
(wide baseline) and writes the frame selection to `demo/demo_index.json`:

```bash
python demo/export_demo.py --min-gap 10 --max-views 12
```

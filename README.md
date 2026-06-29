<div align="center">

# GlobalSplat

### Efficient Feed-Forward 3D Gaussian Splatting via Global Scene Tokens

**ECCV 2026**

Roni Itkin · Noam Issachar · Yehonatan Keypur · Xingyu Chen · Anpei Chen · Sagie Benaim

*The Hebrew University of Jerusalem · Westlake University*

[![arXiv](https://img.shields.io/badge/arXiv-2604.15284-b31b1b.svg)](https://arxiv.org/abs/2604.15284)
[![Project Page](https://img.shields.io/badge/Project-Page-1f72b8.svg)](https://r-itk.github.io/globalsplat/)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-ffce44.svg)](https://huggingface.co/Roni-It/globalsplat)
[![License: PolyForm NC](https://img.shields.io/badge/License-PolyForm%20NC%201.0.0-lightgrey.svg)](./LICENSE)

</div>

> **License:** noncommercial use only (research, academic, personal). Commercial
> use requires a separate license, and military/surveillance use is prohibited.
> See [`LICENSE`](./LICENSE) and [`ADDITIONAL_TERMS.md`](./ADDITIONAL_TERMS.md).

GlobalSplat is a feed-forward, generalizable 3D Gaussian Splatting model built on
the principle **align first, decode later**. Instead of forming primitives in dense
per-view (pixel/voxel-aligned) spaces and reconciling them afterward, GlobalSplat
first fuses all input views into a *compact, fixed-size set of global latent scene
tokens*, then decodes an explicit set of 3D Gaussians from those tokens. The
Gaussian budget is therefore **independent of the number of input views**.

On RealEstate10K with 24 input views it reaches competitive quality (28.5 PSNR at
16K Gaussians; 29.5 at 32K) while using a small fraction of the primitives of dense
baselines — a <4 MB asset, 1.79 GB peak memory, and <78 ms inference.

| RealEstate10K, 24 views | PSNR↑ | SSIM↑ | LPIPS↓ | #Gaussians | Disk |
|---|---|---|---|---|---|
| ZPressor (6 anchors) | 28.51 | 0.911 | 0.097 | 393K | 134 MB |
| DepthSplat | 19.66 | 0.743 | 0.239 | 1572K | 534 MB |
| C3G | 23.80 | 0.747 | 0.198 | 2K | 0.1 MB |
| **GlobalSplat-16K (this repo)** | **28.53** | 0.883 | 0.140 | **16K** | **3.8 MB** |
| GlobalSplat-32K | 29.48 | 0.901 | 0.122 | 32K | ~7.6 MB |

## Contents

- [Quickstart](#quickstart)
- [Method](#method)
- [Repository layout](#repository-layout)
- [Pretrained models](#pretrained-models)
- [Data](#data)
- [Training](#training)
- [Hyperparameters](#hyperparameters)
- [License & citation](#license)

Deeper guides live under [`docs/`](./docs):
[install](./docs/INSTALL.md) ·
[training & data](./docs/TRAINING.md) ·
[evaluation](./docs/EVALUATION.md) ·
[troubleshooting](./docs/TROUBLESHOOTING.md).

## Quickstart

**Requirements:** Linux, Python 3.10, a CUDA 12.x toolkit (`nvcc` on `PATH`), and
an NVIDIA GPU (Ampere+ recommended).

```bash
# Install (auto-detects your GPU arch; compiles gsplat from source).
git clone <this-repo> globalsplat && cd globalsplat
python -m venv .venv && source .venv/bin/activate
bash install.sh

# Sanity-check the install (CPU forward/backward; also renders if a GPU is present).
python -m globalsplat.selfcheck

# Train the main model (point config/dataset/re10k.yaml at your data first).
python -m globalsplat.main +experiment=re10k_16k

# Evaluate a checkpoint on the RE10K 24-view protocol.
python -m globalsplat.main +experiment=re10k_16k dataset=re10k_eval_ctx24 \
    mode=test checkpointing.load=checkpoints/<exp>/last.ckpt
```

Try it without training: [`demo/`](./demo) bundles three example scenes (input
views + cameras + a precomputed reconstruction) you can open in a browser splat
viewer or re-run with `python demo/run_demo.py --scene demo/scenes/<name>`.

Install edge cases (no `nvcc`, mixed-GPU clusters, torch 2.4, perceptual-loss
weights) are in [docs/INSTALL.md](./docs/INSTALL.md). Data download/layout and the
full set of training/resume knobs are in [docs/TRAINING.md](./docs/TRAINING.md);
the evaluation protocol (RE10K + ACID, custom indices) is in
[docs/EVALUATION.md](./docs/EVALUATION.md).

## Method

1. **Scene normalization & input prep** (`globalsplat/dataset/preprocessing.py`,
   `globalsplat/misc/pose_encoding.py`). Cameras are mapped to a canonical frame
   (average camera pose; scale = diameter of the camera constellation,
   YoNoSplat-style). Each view is tokenized into 8×8 patches: patchified RGB tokens
   plus a camera token made of a patchified Plücker-ray embedding and a per-view
   camera code (Fourier features of the camera center + an MLP on the intrinsics).
2. **Dual-branch encoder** (`globalsplat/model/encoder/`). A fixed bank of M = 2048
   learnable latent tokens (dim 512), plus learnable register tokens, is refined by
   B = 4 blocks. Each block runs parallel **geometry** and **appearance** streams:
   the stream queries cross-attend the multi-view token memory, then apply L = 2
   self-attention layers; the two streams are fused by a mixer MLP. This
   disentanglement stops appearance from masking weak geometry.
3. **Dual-branch decoder** (`globalsplat/model/decoder/gaussian_decoder.py`). Two
   heads map the refined tokens to Gaussian geometry (means, log-scales, 6D
   rotation, opacity) and appearance (SH coefficients, degree 3).
4. **Coarse-to-fine capacity curriculum** (paper Sec. 3.4 / App. B). Each token
   predicts a fixed 16 candidates; a geometry-conditioned gate merges them into
   G = 2^s groups, growing s from 0 to 3 over training (schedule 10k/20k/50k steps).
   The final model exposes G = 8, i.e. 2048 × 8 = **16,384** Gaussians. Attribute
   merging is parameter-aware (volume-preserving for log-scale, union-preserving
   for opacity), with smooth linear interpolation across stage transitions.
5. **Training objective** (`globalsplat/loss/`, `globalsplat/model/model_wrapper.py`).
   Rendering loss (MSE + perceptual), a self-supervised **subset-consistency** loss
   (split context into two interleaved subsets sharing anchor views; match rendered
   opacity/depth with a symmetric stop-gradient), and regularizers (soft frustum
   constraint on means + decoder-side scale/rotation/SH terms, plus an optional
   opacity term that is disabled by default — see [Hyperparameters](#hyperparameters)).

Rendering uses [`gsplat`](https://github.com/nerfstudio-project/gsplat). See the
paper for full detail.

## Repository layout

```
config/                 Hydra config tree (model / dataset / loss / trainer) — see config/README.md
globalsplat/
  main.py               entrypoint: train (mode=train) and eval (mode=test)
  selfcheck.py          post-install sanity check (python -m globalsplat.selfcheck)
  checkpointing.py      versioned run dirs + resume resolution
  dataset/              upstream-backed DataModule + our additions
                        (preprocessing.py, data_module.py, re10k.py, dl3dv.py)
  model/
    globalsplat.py      the whole model: tokenize -> slot-encode -> decode
    model_wrapper.py    LightningModule + training/eval logic
    optim.py            optimizer + LR schedule
    rendering.py        gsplat rasterization wrapper
    types.py            Gaussians dataclass (gsplat-shaped output)
    encoder/            dual_stream.py + backbone_layers (vendored VGGT)
    decoder/            gaussian_decoder.py
  loss/                 rendering_loss.py + frustum_loss.py
  misc/                 geometry.py + pose_encoding.py
assets/eval_index/      deterministic eval indices (re10k/, acid/)
third_party/            ZPressor (vendored, trimmed) for datasets + eval utils
demo/                   runnable example scenes (run_demo.py + bundled scenes)
ablate_resolution.py    resolution-scaling ablation (see docs/EVALUATION.md)
tests/                  CPU unit tests (optim, resume, versioning, logging, …)
docs/                   install / training / evaluation / troubleshooting guides
```

## Pretrained models

All weights are on the Hugging Face Hub:
[**Roni-It/globalsplat**](https://huggingface.co/Roni-It/globalsplat).

| Variant | Latents × splats/token | #Gaussians | Disk | RE10K-24 PSNR | Weights |
|---|---|---|---|---|---|
| GlobalSplat-2K  | 2048 × 1 | 2K  | ~0.5 MB | 26.84 | [download](https://huggingface.co/Roni-It/globalsplat/resolve/main/globalsplat-re10k-2k.ckpt) |
| GlobalSplat-16K | 2048 × 8 | 16K | 3.8 MB  | 28.53 | [download](https://huggingface.co/Roni-It/globalsplat/resolve/main/globalsplat-re10k-16k.ckpt) |
| GlobalSplat-16K (no-opacity, this code) | 2048 × 8 | 16K | 3.8 MB | 28.95 | [download](https://huggingface.co/Roni-It/globalsplat/resolve/main/globalsplat-re10k-16k-noopacity.ckpt) |
| GlobalSplat-32K | 4096 × 8 | 32K | ~7.6 MB | 29.48 | [download](https://huggingface.co/Roni-It/globalsplat/resolve/main/globalsplat-re10k-32k.ckpt) |

> The **no-opacity 16K** was trained with the released code (decoder opacity
> regularizer disabled) and outperforms the paper default at every context count
> (12v 29.10 / 24v 28.95 / 36v 28.76 PSNR). The 2K variant uses `M_max=2`
> (`+experiment=re10k_2k`); 16K/32K use the default `M_max=16`.

## Data

Training and the RE10K/ACID evaluation protocols use the preprocessed
**RealEstate10K** and (zero-shot) **ACID** `*.torch` chunks — the same ones used
by pixelSplat / MVSplat / DepthSplat. (The [`demo/`](./demo) needs none of this.)
Acquire them via pixelSplat's
[data instructions](https://github.com/dcharatan/pixelsplat#acquiring-datasets),
or download the preprocessed datasets from Hugging Face
([RE10K](https://huggingface.co/datasets/lhmd/re10k_torch),
[ACID](https://huggingface.co/datasets/lhmd/acid_torch)).

Point each dataset's `dataset_roots` at the folder holding `train/` and `test/`.
The shipped defaults are `../data/re10k/` (`config/dataset/re10k.yaml`) and
`../data/acid/` (`config/dataset/acid.yaml`):

```
../data/
├── re10k/
│   ├── train/
│   │   ├── 000000.torch
│   │   ├── ...
│   │   └── index.json
│   └── test/
│       ├── 000000.torch
│       ├── ...
│       └── index.json
└── acid/                       # zero-shot eval only needs test/
    └── test/
        ├── 000000.torch
        ├── ...
        └── index.json
```

The vendored loaders are already wired (`mvsplat_root:
./third_party/ZPressor/mvsplat`), so only `dataset_roots` needs editing. Full
data + resume details are in [docs/TRAINING.md](./docs/TRAINING.md).

## Training

GlobalSplat-16K is the main released model: 2048 latents × 8 splats/token = 16K
Gaussians (`M_max=16`, `curriculum.final_stage=3`).

```bash
# Train the 16K model
python -m globalsplat.main +experiment=re10k_16k

# Evaluate a checkpoint on the fixed RE10K 24-view protocol (deterministic index)
python -m globalsplat.main +experiment=re10k_16k dataset=re10k_eval_ctx24 \
    mode=test checkpointing.load=<ckpt>
```

The Gaussian budget is `model.latent_rep_token_amount × 2**curriculum.final_stage`;
the other budgets are selectable from the command line via their experiment
configs (`+experiment=re10k_2k` / `re10k_32k`). ACID cross-dataset (zero-shot)
reuses the RE10K-trained weights with `dataset=acid_eval_ctx{12,24,36}` — see
[docs/EVALUATION.md](./docs/EVALUATION.md).

## Hyperparameters

The shipped configs reproduce the paper's settings (App. B / Table 6): SH degree 3,
6D rotation, mean offset (0,0,1.5), log-scale offset -2, opacity-logit offset -5;
2048 latents, dim 512, patch 8, 4 encoder blocks, 2 self-attention layers, 16
candidates per token (`M_max`; the 2K variant uses `M_max=2`), stage-3
(8 Gaussians/token); AdamW lr 5e-4, weight decay 1e-6,
gradient clip 1.0, linear warm-up + cosine; loss weights λmse=2, λperc=1, λfru=1e-2,
λdec=1e-2, λα_con=1e-3, λd_con=1e-2.

> **Opacity regularizer.** The released checkpoint's training run disables the
> decoder opacity regularizer (paper App. B, Eq. 44). To match the full paper
> objective, re-enable the opacity term in
> `globalsplat/model/decoder/gaussian_decoder.py`.

## Notes on this release

- Transformer layer blocks are vendored from VGGT under
  `globalsplat/model/encoder/backbone_layers/`; ZPressor is vendored under
  `third_party/`. See [`NOTICE.md`](./NOTICE.md).

## License

GlobalSplat's own code is licensed under the **PolyForm Noncommercial License
1.0.0** ([`LICENSE`](./LICENSE)), subject to [`ADDITIONAL_TERMS.md`](./ADDITIONAL_TERMS.md),
which reserve all commercial use for separate written licensing and prohibit
military and surveillance use. This is a source-available, noncommercial license —
not an OSI open-source license.

- **Noncommercial research, academic, and personal use** is permitted.
- **Commercial use** requires a separate written license — contact roni.itkin@gmail.com.
- **Military and surveillance use** is prohibited (`ADDITIONAL_TERMS.md` §2).

Vendored and third-party components (VGGT, ZPressor/MVSplat/DepthSplat/pixelSplat,
and external dependencies such as `gsplat` and PyTorch) retain their own licenses;
see [`NOTICE.md`](./NOTICE.md).

## Citation

```bibtex
@article{itkin2026globalsplat,
  title   = {GlobalSplat: Efficient Feed-Forward 3D Gaussian Splatting via Global Scene Tokens},
  author  = {Itkin, Roni and Issachar, Noam and Keypur, Yehonatan and Chen, Xingyu and Chen, Anpei and Benaim, Sagie},
  journal = {arXiv preprint arXiv:2604.15284},
  year    = {2026},
  note    = {To appear at ECCV 2026}
}
```

## Acknowledgements

Built on [ZPressor](https://github.com/ziplab/ZPressor),
[MVSplat](https://github.com/donydchen/mvsplat),
[DepthSplat](https://github.com/cvg/depthsplat),
[pixelSplat](https://github.com/dcharatan/pixelsplat),
[VGGT](https://github.com/facebookresearch/vggt), and
[gsplat](https://github.com/nerfstudio-project/gsplat). We acknowledge EuroHPC JU
(project EHPC-AIF-2025SC02-060, Leonardo @ CINECA) and the ISF (grant 2416/25).

# Evaluation

Evaluation uses the same API as the ZPressor/MVSplat/DepthSplat family: set
`mode=test` and point `checkpointing.load` at a checkpoint. It reuses the upstream
`compute_psnr/ssim/lpips`, `Benchmarker`, and image IO, and writes `scores_*.json`
+ `benchmark.json` under `test.output_path/<experiment_name>`. Toggle
`test.save_image`, `test.save_gt_image`, `test.save_video`.

PSNR / SSIM / LPIPS are the reported metrics.

## RealEstate10K (paper protocol)

The paper follows the C3G RealEstate10K protocol (built on the NoPoSplat split)
at 256×256: a fixed set of held-out targets is rendered from 12 / 24 / 36 context
views. Those fixed `{scene: {context, target}}` indices ship under
`assets/eval_index/re10k/` and are wired as ready-to-use dataset configs
(`re10k_eval_ctx12` / `ctx24` / `ctx36`), which swap the random `bounded_v2`
sampler for the deterministic `evaluation` sampler:

```bash
# 24 context views, fixed targets
python -m globalsplat.main +experiment=re10k_16k dataset=re10k_eval_ctx24 \
    mode=test checkpointing.load=checkpoints/<exp>/last.ckpt
```

Swap `re10k_eval_ctx24` for `ctx12` / `ctx36` for the other context counts. Any
checkpoint works at any context count (the model is feed-forward over the context
set).

## ACID (zero-shot cross-dataset)

ACID is evaluated zero-shot with the RE10K-trained weights. ACID ships in the
same `*.torch` chunk format as RE10K, so it reuses the RE10K loader unchanged;
only the data roots and the eval index differ. The fixed C3G ACID indices
(12 / 24 / 36 context views, 3 targets) ship under `assets/eval_index/acid/` and
are wired as ready-to-use dataset configs (`acid_eval_ctx12` / `ctx24` / `ctx36`).

First point `config/dataset/acid.yaml: dataset_roots` at your ACID chunks (the
folder with `test/` holding `*.torch` + `index.json`), then:

```bash
# ACID, 24 context views, fixed targets, RE10K-trained checkpoint
python -m globalsplat.main +experiment=re10k_16k dataset=acid_eval_ctx24 \
    mode=test checkpointing.load=checkpoints/<exp>/last.ckpt
```

Swap `acid_eval_ctx24` for `ctx12` / `ctx36` for the other context counts. This
reproduces the ACID rows of the paper's cross-dataset table.

## Custom indices

To use a custom index, set `dataset.eval_index_path=<file.json>` and
`dataset.num_context_views=<N>` directly. The index is a JSON map
`{scene: {context: [...], target: [...]}}`; the test progress-bar total is taken
from the number of scenes in this index.

## High-resolution input & resolution-scaling ablation

GlobalSplat is trained at 256². Because the Gaussian budget is fixed, it scales
to higher-resolution **input** far more cheaply than dense baselines — on
RealEstate10K (12 views, Ours-32K):

| Res. | Method | #G | Time | Mem (GB) | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|
| 256² | DepthSplat | 786K | 289 ms | 23.6 | 21.35 | 0.809 | 0.190 |
| 256² | ZPressor | 393K | 169 ms | 3.6 | 28.46 | 0.910 | 0.098 |
| 256² | **Ours-32K** | **32K** | **104 ms** | **1.8** | **29.54** | 0.903 | 0.121 |
| 512² | DepthSplat | 3145K | 1301 ms | 58.2 | 21.84 | 0.815 | 0.261 |
| 512² | ZPressor | 1572K | 852 ms | 18.5 | 28.11 | 0.900 | 0.160 |
| 512² | **Ours-32K** | **32K** | **111 ms** | **3.4** | **29.08** | 0.886 | 0.206 |

Naively feeding a 512² input degrades sharply because the camera-ray (Plücker)
embedding is **out-of-distribution** at the new resolution — not the render
resolution, Gaussian count, or token count. We fix this with a **polyphase
virtual-camera** scheme: the 512² context is split into `phase²=4` interlaced 256²
sub-images (subsampling every other pixel per axis), each sub-camera's principal
point shifted so its rays coincide exactly with the original pixel rays. The
Plücker tokens stay in-distribution while all 512² pixels are kept.

`ablate_resolution.py` reproduces the controlled ablation (RealEstate10K, 12
views, model trained at 256²; `CK` = the 32K weights, `IDX` = the ctx12 index):

| Setting | Input | Render | Plücker | PSNR | SSIM | LPIPS | `--mode` |
|---|---|---|---|---|---|---|---|
| Naive high-res | 512² | 512² | OOD | 26.16 | 0.807 | 0.276 | `plain --resolution 512` |
| Render-only | 256² | 512² | in-dist. | 28.90 | 0.881 | 0.210 | `render-only --resolution 512 --context-resolution 256` |
| Duplicated tokens | 256²×4 | 256² | in-dist. | 29.54 | 0.904 | 0.121 | `duplicate --resolution 256 --duplicate 4` |
| Virtual cameras (ours) | 512²→4×256² | 512² | in-dist. | 29.08 | 0.886 | 0.206 | `virtual --resolution 512 --phase 2` |

```bash
CK=globalsplat-re10k-32k.ckpt
IDX=assets/eval_index/re10k/c3g_re10k_ctx_12v_trg_3v.json
python ablate_resolution.py --ckpt $CK --mode virtual --resolution 512 --phase 2 \
    --overrides dataset.eval_index_path=$IDX
```

The render-only and duplicated-token controls degrade only mildly, isolating the
large naive-512² drop to the input rays. (The duplicated-token control renders at
256², so its higher PSNR is expected; among 512²-render settings, the polyphase
virtual-camera scheme is best.)

# Configuration (Hydra)

GlobalSplat is configured with [Hydra](https://hydra.cc). `main.yaml` is the root
config: it composes one file from each group (`model`, `dataset`, `loss`,
`trainer`) and holds the run-level settings (`mode`, `optimizer`, `checkpointing`,
`curriculum`, `test`). Every leaf is overridable on the command line.

## Layout

```
config/
  main.yaml            root: composes the groups + mode / optimizer / checkpointing / curriculum / test
  model/
    globalsplat.yaml   architecture: latents, dim, patch size, encoder rounds, M_max, ...
  dataset/
    re10k.yaml         RealEstate10K (train + bounded-sampler eval) — set `dataset_roots`
    acid.yaml          ACID via the RE10K loader (zero-shot cross-dataset)
    dl3dv.yaml         DL3DV (DepthSplat loader)
    re10k_eval_ctx{12,24,36}.yaml   deterministic RE10K eval (fixed index, 3 targets)
    acid_eval_ctx{12,24,36}.yaml    deterministic ACID eval
  loss/
    default.yaml       loss weights (MSE / perceptual / frustum / subset-consistency / ...)
  trainer/
    default.yaml       Lightning Trainer: devices, precision, max_steps, ...
  experiment/
    re10k_2k.yaml      `+experiment=` overlays: pick a dataset + the Gaussian
    re10k_16k.yaml     budget (latents × 2**final_stage) + the experiment name
    re10k_32k.yaml
    dl3dv.yaml
```

## How it composes

`main.yaml`'s `defaults` select `model=globalsplat`, `dataset=re10k`,
`loss=default`, `trainer=default`. An `+experiment=` overlay (`@package _global_`)
then overrides the dataset group and a few knobs. The headline knob is the
**Gaussian budget**:

```
#Gaussians = model.latent_rep_token_amount × 2**curriculum.final_stage
```

| Experiment   | latents | final_stage | M_max | #Gaussians |
|---|---|---|---|---|
| `re10k_2k`   | 2048 | 0 | 2  | 2K  |
| `re10k_16k`  | 2048 | 3 | 16 | 16K |
| `re10k_32k`  | 4096 | 3 | 16 | 32K |

(`M_max` = candidate splats predicted per token before coarse-to-fine gating;
`final_stage` must satisfy `2**final_stage <= M_max`.)

## Using it

```bash
# train with an experiment overlay
python -m globalsplat.main +experiment=re10k_16k

# deterministic eval: swap the dataset group for an eval overlay
python -m globalsplat.main +experiment=re10k_16k dataset=re10k_eval_ctx24 \
    mode=test checkpointing.load=<ckpt>

# override any leaf from the CLI (dotted path)
python -m globalsplat.main +experiment=re10k_16k optimizer.lr=3e-4 trainer.devices=1
```

Before training/eval, point the dataset config's `dataset_roots` at your `*.torch`
chunks — see the top-level [README "Data"](../README.md#data). Checkpointing /
resume knobs (`checkpointing.load` / `resume` / `auto_resume` / `exact_resume`)
are documented in [docs/TRAINING.md](../docs/TRAINING.md).

# Training & Data

## Data

GlobalSplat consumes **RealEstate10K** (train + eval) and **ACID** (zero-shot
cross-dataset) through the upstream loaders, so use the **same preprocessed
`*.torch` chunks** the ZPressor / DepthSplat / pixelSplat repos use. Each dataset
root holds `train/` and `test/` subfolders, each with `*.torch` chunks plus an
`index.json`.

- **RE10K**: point `config/dataset/re10k.yaml: dataset_roots` at the folder that
  contains `train/` and `test/`.
- **ACID**: point `config/dataset/acid.yaml: dataset_roots` at the ACID chunks
  (zero-shot eval only needs `test/`). See [EVALUATION.md](./EVALUATION.md).
- **DL3DV**: set `config/dataset/dl3dv.yaml: depthsplat_root` and `dataset_roots`.

> **Getting the chunks.** Acquire the preprocessed `*.torch` chunks the same way
> as pixelSplat / DepthSplat: follow pixelSplat's
> [data instructions](https://github.com/dcharatan/pixelsplat#acquiring-datasets),
> or download the preprocessed datasets from Hugging Face
> ([RE10K](https://huggingface.co/datasets/lhmd/re10k_torch),
> [ACID](https://huggingface.co/datasets/lhmd/acid_torch)).

Expected folder structure (point each config's `dataset_roots` at the dataset folder):

```
datasets/
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

Scene normalization, pixel-intrinsics conversion, and the `{context,target}` →
`{inputs,targets}` rename are the **only** additions on top of the upstream
pipeline; they live in `globalsplat/dataset/preprocessing.py`. Chunk loading, the
bounded view sampler, augmentation, cropping, the DataLoader, seeding, and
collation are the upstream MVSplat/DepthSplat code, driven unchanged through their
`DataModule` (`globalsplat/dataset/data_module.py`). The package is named
`globalsplat` (not `src`) precisely so it can import the upstream `src.*` modules
without a name clash.

## Train

```bash
python -m globalsplat.main +experiment=re10k_16k             # main model
python -m globalsplat.main +experiment=dl3dv                 # switch dataset
python -m globalsplat.main +experiment=re10k_16k trainer.devices=1 optimizer.batch_size=1
```

Hardware/schedule knobs are in `config/trainer/default.yaml` (`devices`,
`num_nodes`, `max_steps`, `precision`, …). Checkpoints go to
`checkpoints/<experiment_name>/` and TensorBoard logs (scalars + GT-vs-pred
videos) go to `logs/`.

## Attention & precision

Both attention paths (the Perceiver cross-attention and the VGGT-style block
self-attention) use `F.scaled_dot_product_attention`, which PyTorch dispatches to
**FlashAttention-2** on Ampere+ GPUs under bf16/fp16 -- no `flash_attn` package
needed. The entrypoint enables the flash SDP backend (with mem-efficient/math as
fallbacks). Precision defaults to fp32 (`precision=32`, as released); to use the
FlashAttention kernel, opt in with `trainer.precision=bf16-mixed`. That override
is safe: the ray/Plücker geometry and the gsplat rasterizer always run in fp32
internally regardless of the autocast precision.

## Checkpointing & resume

Each run gets its own versioned directory: checkpoints go to
`<output_dir or ./checkpoints>/<experiment_name>/version_<N>/` (with `last.ckpt`)
and TensorBoard logs to `logs/<experiment_name>/version_<N>/`, sharing the same
`version_<N>`, so re-running the same `experiment_name` never overwrites a
previous run.

Resume priority:

- `checkpointing.load` + `resume=true` resumes full optimizer/step state **from
  that exact path**, continuing in that checkpoint's own `version_<N>` directory
  (so preemption truly continues in place).
- `load` + `resume=false` loads **weights only** (fine-tune) into a fresh version,
  leaving the source run untouched.
- otherwise, if `checkpointing.auto_resume=true` (default), training resumes the
  latest `version_<N>/last.ckpt` found under the experiment dir, in place -- so
  preempted/SLURM jobs continue seamlessly on restart.

The **data schedule resumes too**: the upstream `StepTracker` step (which drives
the bounded view-sampler warm-up) is checkpointed and restored, and the train
shuffle seed is keyed on the epoch and the resumed step so the data stream
advances rather than replaying from the start. (The RE10K/DL3DV loaders are
streaming `IterableDataset`s, so within an in-progress epoch resumption continues
the schedule and advances the shuffle stream rather than restoring an exact
byte-offset into the chunk stream.)

For **exact mid-epoch resume**, set `checkpointing.exact_resume=true`: the train
loader becomes stateful (implements `state_dict`/`load_state_dict`, so Lightning
recognizes it and the "dataloader is not resumable" warning goes away) and
fast-forwards past the batches already consumed in the interrupted epoch, paired
with a per-epoch-deterministic shuffle (and `persistent_workers=False` +
`reload_dataloaders_every_n_epochs=1`) so the replayed order is identical and you
land on the exact same samples. The cost is a one-time re-stream of the consumed
chunks on resume; leave it off (default) for the cheaper "advance to a fresh
shuffle" behavior.

## Training view counts (the 24-context / 13-view recipe)

The shipped `config/dataset/re10k.yaml` uses **24 context / 12 target** views per
sample. With the subset-consistency objective active, the context set is split
into two subsets -- the two min/max frames are shared anchors and the middle views
alternate -- and each subset gets its *own* forward pass; the model never fuses the
full context at once during training.

So 24 context yields **13-view forward passes** (2 anchors + 11 middle), which is
exactly the paper's "13 input views" (App. B.2 / Table 6), and the 12 shared
targets match as well -- i.e. 24/12 reproduces the paper recipe. This per-forward
subset size is the key knob for 12 / 24 / 36-view eval generalization: the earlier
12-context config produced only 7-view forwards and regressed at 24/36.

> **Note:** do *not* set `num_context_views: 13` to "match the paper" -- that
> yields 8-view forwards. **24** is the value that reproduces the paper's 13-view
> passes.

Scale `trainer.devices` / `accumulate_grad_batches` to hold the global batch size
at 16.

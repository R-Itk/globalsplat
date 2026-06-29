"""Training / evaluation entrypoint.

Mirrors the ZPressor/MVSplat/DepthSplat API: a single Hydra entrypoint with a
``mode`` switch (``train`` -> ``trainer.fit``, ``test`` -> ``trainer.test``) and
``checkpointing.load`` / ``checkpointing.resume``.

    python -m globalsplat.main +experiment=re10k_16k
    python -m globalsplat.main +experiment=re10k_16k mode=test \
        checkpointing.load=/path/to/last.ckpt
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import warnings
from pathlib import Path

# The vendored upstream loads its .torch chunks with torch.load(weights_only=False)
# (they hold pickled metadata, not just tensors). That is expected; silence the noisy
# per-chunk FutureWarning rather than touch upstream code.
warnings.filterwarnings("ignore", message=r".*weights_only=False.*", category=FutureWarning)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


try:  # print only from the main DDP process so logs aren't duplicated per rank
    from lightning_fabric.utilities.rank_zero import rank_zero_only as _rank_zero_only
except Exception:  # pragma: no cover
    from pytorch_lightning.utilities.rank_zero import rank_zero_only as _rank_zero_only


@_rank_zero_only
def rank_zero_print(*args, **kwargs) -> None:
    print(*args, **kwargs)


def _quiet_upstream_logs(cfg: DictConfig) -> None:
    """Optionally drop the upstream loader's per-example "Skipped bad example ..."
    spam (a plain ``print`` inside the vendored dataset). Patches ``builtins.print``
    to filter only that exact prefix; everything else passes through. Forked
    DataLoader workers inherit this, so their prints are filtered too."""
    if not bool(cfg.get("quiet_skipped_examples", True)):
        return
    import builtins

    _orig = builtins.print

    def _filtered(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("Skipped bad example"):
            return
        return _orig(*args, **kwargs)

    builtins.print = _filtered


def enable_flash_attention() -> None:
    """Prefer the FlashAttention SDPA kernel for all attention.

    Both attention paths (the slot-encoder cross-attention and the VGGT-style block
    self-attention) call ``F.scaled_dot_product_attention``; PyTorch dispatches
    that to FlashAttention-2 on Ampere+ GPUs under bf16/fp16. We enable flash and
    keep mem-efficient + math as fallbacks (e.g. on CPU or for ineligible shapes).
    No ``flash_attn`` package is required.
    """
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


enable_flash_attention()

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from globalsplat.model.globalsplat import GlobalSplat
from globalsplat.model.model_wrapper import GlobalSplatModule
from globalsplat.checkpointing import resolve_run


class GsplatWarmupCallback(pl.Callback):
    """Build gsplat's CUDA extension once before any rank rasterizes.

    gsplat JIT-compiles its CUDA backend on first use. Under multi-GPU DDP on a
    networked filesystem, several ranks racing to build/clean the same build dir
    triggers ``OSError: [Errno 16] Device or resource busy: .nfsXXXX``. Here rank 0
    triggers the build, everyone waits on a barrier, then the other ranks load the
    (now cached) extension — so no two ranks ever build/clean concurrently.

    For extra safety on NFS clusters, also point the build cache at node-local
    scratch, e.g. ``export TORCH_EXTENSIONS_DIR=$TMPDIR/torch_ext``.
    """

    @staticmethod
    def _load_backend() -> None:
        try:
            from gsplat.cuda._backend import _C  # noqa: F401  (forces the build)
        except Exception as exc:  # pragma: no cover - depends on CUDA toolchain
            rank_zero_print(f"[gsplat warmup] could not pre-build CUDA extension: {exc}")

    def on_fit_start(self, trainer, pl_module) -> None:
        if not torch.cuda.is_available():
            return
        if trainer.is_global_zero:
            self._load_backend()
        trainer.strategy.barrier()
        if not trainer.is_global_zero:
            self._load_backend()
        trainer.strategy.barrier()

_NON_MODEL_KWARGS = {"name"}


def build_model(model_cfg: DictConfig) -> GlobalSplat:
    kwargs = {
        k: v
        for k, v in OmegaConf.to_container(model_cfg, resolve=True).items()
        if k not in _NON_MODEL_KWARGS
    }
    return GlobalSplat(**kwargs)


def _eval_index_len(cfg: DictConfig, is_test: bool) -> int | None:
    """Scene count of the deterministic eval index, used to set an accurate test
    progress-bar total.

    The upstream IterableDataset's ``__len__`` counts *every* scene in the loaded
    chunks (e.g. the full ~7289-scene RE10K test set), but the evaluation view
    sampler only emits scenes that are present in the eval index -- all others are
    skipped. Without this, the test bar shows the chunk-set total and overshoots
    the number of scenes actually evaluated (e.g. 5601 for RE10K, 1341 for ACID).

    Returns the index size (to pass as ``limit_test_batches``, since the test
    loader uses batch size 1) or ``None`` to leave Lightning's default.
    """
    if not is_test:
        return None
    idx = cfg.dataset.get("eval_index_path", None)
    if not idx:
        return None
    # Resolve a relative index path the same way build_re10k does: prefer the
    # path as given (cwd), then fall back to the repo root.
    p = Path(idx).expanduser()
    if not p.is_absolute() and not p.exists():
        repo_root = Path(__file__).resolve().parents[1]
        if (repo_root / p).exists():
            p = repo_root / p
    try:
        with open(p) as f:
            return len(json.load(f)) or None
    except Exception:
        # Cosmetic only -- never fail eval over the progress-bar total.
        return None


def build_datamodule(cfg: DictConfig):
    ds = cfg.dataset
    opt = cfg.optimizer
    common = dict(
        dataset_roots=list(ds.dataset_roots),
        image_shape=tuple(ds.image_shape),
        num_context_views=ds.num_context_views,
        num_target_views=ds.num_target_views,
        min_distance_between_context_views=ds.min_distance_between_context_views,
        max_distance_between_context_views=ds.max_distance_between_context_views,
        initial_min_distance_between_context_views=ds.initial_min_distance_between_context_views,
        initial_max_distance_between_context_views=ds.initial_max_distance_between_context_views,
        context_gap_warm_up_steps=ds.context_gap_warm_up_steps,
        max_distance_to_context_views=ds.max_distance_to_context_views,
        target_gap_warm_up_steps=ds.get("target_gap_warm_up_steps", 0),
        initial_max_distance_to_context_views=ds.get("initial_max_distance_to_context_views", 0),
        extra_views_sampling_strategy=ds.extra_views_sampling_strategy,
        target_views_replace_sample=ds.target_views_replace_sample,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        seed=cfg.seed,
        pin_memory=ds.get("pin_memory", True),
        prefetch_factor=opt.get("prefetch_factor", 2),
        scene_scale_factor=ds.get("scene_scale_factor", 1.0),
        # Exact mid-epoch resume: stateful, fast-forwarding train loader (see
        # ResumableDataLoader / config checkpointing.exact_resume).
        resumable_loader=cfg.checkpointing.get("exact_resume", False),
        # Optional fixed eval index (RE10K). When set, the test/eval dataloader
        # uses the deterministic "evaluation" view sampler instead of bounded_v2.
        eval_index_path=ds.get("eval_index_path", None),
    )

    if ds.name == "re10k":
        from globalsplat.dataset.re10k import ZPressorRE10KDataModule, ZPressorStepTrackerCallback
        dm = ZPressorRE10KDataModule(
            mvsplat_root=ds.mvsplat_root,
            near=ds.near, far=ds.far,
            make_baseline_1=ds.make_baseline_1,
            baseline_scale_bounds=ds.baseline_scale_bounds,
            augment=ds.augment,
            **common,
        )
        return dm, ZPressorStepTrackerCallback()

    if ds.name == "dl3dv":
        from globalsplat.dataset.dl3dv import DepthSplatDL3DVDataModule, DepthSplatStepTrackerCallback
        dm = DepthSplatDL3DVDataModule(
            depthsplat_root=ds.depthsplat_root,
            packed_image_shape=tuple(ds.packed_image_shape),
            **common,
        )
        return dm, DepthSplatStepTrackerCallback()

    raise ValueError(f"Unknown dataset.name={ds.name!r}")


def build_module(cfg: DictConfig, eval_mode: bool) -> GlobalSplatModule:
    model = build_model(cfg.model)
    return GlobalSplatModule(
        model,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.get("weight_decay", 0.0),
        loss_w_dict=OmegaConf.to_container(cfg.loss, resolve=True),
        batch_size=cfg.optimizer.batch_size,
        eval_mode=eval_mode,
        stage_boundaries=tuple(cfg.curriculum.boundaries),
        stage_ramp_iters=cfg.curriculum.ramp_iters,
        final_stage=cfg.curriculum.final_stage,
        test_cfg=OmegaConf.to_container(cfg.test, resolve=True),
        upstream_repo_root=cfg.dataset.get("mvsplat_root", cfg.dataset.get("depthsplat_root")),
        experiment_name=cfg.experiment_name,
        warmup_pct=cfg.optimizer.get("warmup_pct", 0.03),
        min_lr_ratio=cfg.optimizer.get("min_lr_ratio", 1.0 / 50.0),
        min_lr_floor=cfg.optimizer.get("min_lr_floor", 1e-6),
        warmup_epochs_fallback=cfg.optimizer.get("warmup_epochs_fallback", 5),
    )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    _quiet_upstream_logs(cfg)
    rank_zero_print(OmegaConf.to_yaml(cfg))

    is_test = str(cfg.mode) == "test"
    if is_test:
        # cudnn.benchmark (enabled at import for training speed) autotunes conv
        # algorithms on the first forward, allocating large scratch workspaces.
        # At eval the first metric (VGG/LPIPS) forward triggers a one-off multi-GB
        # spike that gets recorded as the run's peak. Turn it off for eval; the
        # conv-speed difference is negligible here.
        torch.backends.cudnn.benchmark = False
    lit = build_module(cfg, eval_mode=is_test)
    datamodule, step_tracker_cb = build_datamodule(cfg)

    # Versioned run dirs: each run lands in <root>/<exp>/version_<N>/ (checkpoints)
    # and logs/<exp>/version_<N>/ (TensorBoard), so re-running the same experiment
    # never clobbers a previous run. resolve_run() picks the version and the
    # resume target (see globalsplat/checkpointing.py).
    ckpt_root = cfg.output_dir or "./checkpoints"
    exp_ckpt_base = os.path.join(ckpt_root, cfg.experiment_name)
    log_exp_base = os.path.join(cfg.log_dir, cfg.experiment_name)

    run = None
    if not is_test:
        run = resolve_run(
            load=cfg.checkpointing.get("load", None),
            resume=cfg.checkpointing.get("resume", False),
            auto_resume=cfg.checkpointing.get("auto_resume", True),
            exp_ckpt_base=exp_ckpt_base,
            log_exp_base=log_exp_base,
        )

    logger = TensorBoardLogger(
        save_dir=cfg.log_dir,
        name=cfg.experiment_name,
        version=(run.version if run is not None else None),
        default_hp_metric=False,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=(run.ckpt_dir if run is not None else exp_ckpt_base),
        filename="step{step:09d}",
        auto_insert_metric_name=False,
        save_last=True,
        save_top_k=cfg.checkpointing.save_top_k,
        every_n_train_steps=cfg.checkpointing.every_n_train_steps,
    )

    tr = cfg.trainer
    trainer = pl.Trainer(
        logger=logger,
        strategy=tr.strategy,
        use_distributed_sampler=False,
        accelerator=tr.accelerator,
        devices=(1 if is_test else tr.devices),
        num_nodes=(1 if is_test else tr.num_nodes),
        log_every_n_steps=tr.log_every_n_steps,
        accumulate_grad_batches=tr.accumulate_grad_batches,
        gradient_clip_val=tr.gradient_clip_val,
        inference_mode=False,
        max_steps=tr.max_steps,
        max_epochs=-1,
        enable_checkpointing=not is_test,
        callbacks=[step_tracker_cb] + ([] if is_test else [ckpt_cb, GsplatWarmupCallback()]),
        check_val_every_n_epoch=tr.check_val_every_n_epoch,
        limit_val_batches=tr.limit_val_batches,
        # Make the test progress-bar total match the scenes actually evaluated
        # (the eval index), not the full chunk set the upstream __len__ reports.
        limit_test_batches=(_eval_index_len(cfg, is_test) or 1.0),
        num_sanity_val_steps=tr.num_sanity_val_steps,
        precision=tr.precision,
        # Exact mid-epoch resume needs each epoch's loader rebuilt so its
        # per-epoch-deterministic shuffle seed is re-applied (otherwise a single
        # long-lived loader's RNG drifts and the resumed order wouldn't match).
        reload_dataloaders_every_n_epochs=(
            1 if (not is_test and cfg.checkpointing.get("exact_resume", False)) else 0
        ),
    )

    if is_test:
        if not cfg.checkpointing.load:
            raise ValueError("mode=test requires checkpointing.load=<path/to/.ckpt>.")
        lit.model.set_stage(cfg.curriculum.final_stage, mix=1.0)
        trainer.test(lit, datamodule=datamodule, ckpt_path=cfg.checkpointing.load)
        return

    # Training: resolve resume vs weights-only load (version-aware).
    if run.weights_only is not None:
        state = torch.load(run.weights_only, map_location="cpu")
        missing, unexpected = lit.load_state_dict(state.get("state_dict", state), strict=False)
        rank_zero_print(f"Loaded weights from {run.weights_only} (no optimizer/step state). "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    if run.resume_ckpt is not None:
        rank_zero_print(f"Resuming full training state from {run.resume_ckpt} "
              f"(version_{run.version}; checkpoints -> {run.ckpt_dir}).")
    else:
        rank_zero_print(f"Starting training from scratch "
              f"(version_{run.version}; checkpoints -> {run.ckpt_dir}).")
    trainer.fit(lit, datamodule=datamodule, ckpt_path=run.resume_ckpt)


if __name__ == "__main__":
    main()

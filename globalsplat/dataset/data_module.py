"""Drive the upstream (ZPressor/MVSplat, DepthSplat) data pipeline directly.

The upstream ``DataModule`` (chunk loading, view sampling, augmentation, crop,
DataLoader, seeding, collation) is used unchanged. We only wrap its dataloaders
to apply the three additions in ``preprocessing.py`` (scene normalization,
pixel intrinsics, ``{context,target}`` -> ``{inputs,targets}`` rename), and we
update the upstream shared-memory ``StepTracker`` each step so the bounded view
sampler's context-gap warm-up keeps working.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .preprocessing import to_model_batch

try:  # print the resume message once (rank 0), not per DDP rank
    from lightning_fabric.utilities.rank_zero import rank_zero_only as _rank_zero_only
except Exception:  # pragma: no cover
    from pytorch_lightning.utilities.rank_zero import rank_zero_only as _rank_zero_only


@_rank_zero_only
def _rank_zero_print(*args, **kwargs) -> None:
    print(*args, **kwargs)


class ResumableDataLoader(DataLoader):
    """Train DataLoader that PyTorch Lightning recognizes as *stateful*.

    It implements ``state_dict``/``load_state_dict`` (so it satisfies Lightning's
    ``_Stateful`` protocol) and fast-forwards past the batches already consumed in
    the current epoch when resuming. This makes mid-epoch resumption continue at
    the exact same position instead of restarting the epoch -- and silences the
    "your dataloader is not resumable" warning.

    Exactness note: landing on the *same samples* requires the epoch's shuffle
    order to match the interrupted run. ``UpstreamBackedDataModule`` pairs this
    loader with a per-epoch-deterministic seed and ``persistent_workers=False`` so
    re-streaming the consumed prefix reproduces the original order; the
    fast-forward then discards exactly those already-seen batches. The trade-off is
    a one-time re-stream of the consumed chunks on resume.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batches_seen = 0  # batches yielded so far in the current epoch
        self._skip = 0          # set by load_state_dict; consumed on the next __iter__

    def state_dict(self) -> dict:
        return {"batches_seen": int(self._batches_seen)}

    def load_state_dict(self, state_dict: dict) -> None:
        self._skip = int(state_dict.get("batches_seen", 0))

    def __iter__(self):
        it = super().__iter__()
        skip, self._skip = self._skip, 0  # only fast-forward once, on the resumed epoch
        seen = 0
        if skip > 0:
            _rank_zero_print(
                f">> ResumableDataLoader: fast-forwarding {skip} batches to resume "
                f"mid-epoch (re-streaming consumed chunks; one-time cost)...",
                flush=True,
            )
            for _ in range(skip):
                try:
                    next(it)
                except StopIteration:
                    break
                seen += 1
        self._batches_seen = seen  # includes the skipped batches, so a re-save is correct
        for batch in it:
            self._batches_seen += 1
            yield batch


def add_repo_to_path(repo_root: str | Path, *, repo_name: str) -> Path:
    root = Path(repo_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"{repo_name} root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class UpstreamBackedDataModule(pl.LightningDataModule):
    """LightningDataModule that drives the upstream dataset/sampler/shim but builds
    the DataLoaders itself so that:
      * the {context,target}->{inputs,targets} transform (incl. normalization) runs
        inside the worker ``collate_fn`` -- prefetched and overlapped with the GPU,
        not serially in the main process; and
      * ``pin_memory`` / ``prefetch_factor`` are set (the upstream DataModule omits
        both), matching the original training throughput.
    """

    def __init__(
        self,
        upstream_dm,
        step_tracker,
        scene_scale_factor: float = 1.0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        resumable_loader: bool = False,
    ):
        super().__init__()
        self._dm = upstream_dm
        self.step_tracker = step_tracker
        self._ssf = float(scene_scale_factor)
        self._pin_memory = bool(pin_memory)
        self._prefetch_factor = prefetch_factor
        # Opt-in exact mid-epoch resume: use a stateful, fast-forwarding train
        # loader + per-epoch-deterministic shuffle (see ResumableDataLoader).
        self._resumable_loader = bool(resumable_loader)
        # Step the run is resuming from (0 = fresh). Restored from the checkpoint
        # via ``load_state_dict`` so the data schedule (StepTracker warm-up + the
        # train shuffle stream) continues instead of restarting from step 0.
        self._resumed_step = 0

    # ------------------------------------------------------------------
    # Resume support: checkpoint the upstream StepTracker so the bounded
    # view-sampler warm-up and the train shuffle stream pick up where they
    # left off. Lightning saves ``state_dict()`` into the checkpoint and calls
    # ``load_state_dict()`` on resume (before the first batch).
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        step = 0
        if self.step_tracker is not None:
            try:
                step = int(self.step_tracker.get_step())
            except Exception:
                step = 0
        return {"step_tracker_step": step}

    def load_state_dict(self, state: dict) -> None:
        step = int(state.get("step_tracker_step", 0))
        self._resumed_step = step
        # Apply immediately so the very first resumed batch samples views at the
        # restored step rather than at 0 (the per-step callback only fires once
        # the first train batch starts).
        if self.step_tracker is not None and step > 0:
            try:
                self.step_tracker.set_step(step)
            except Exception:
                pass

    @staticmethod
    def _train_seed(base_seed: int, global_rank: int, epoch: int, resumed_step: int) -> int:
        # Make the train shuffle order depend on the epoch (so each epoch is a
        # distinct, reproducible permutation) and on the resumed step (so a run
        # resumed mid-stream advances to a fresh shuffle instead of replaying the
        # already-consumed prefix). 0x9E3779B1 decorrelates consecutive epochs.
        return int(base_seed) + int(global_rank) + 0x9E3779B1 * int(epoch) + int(resumed_step)

    def _collate_fn(self):
        from torch.utils.data import default_collate

        ssf = self._ssf

        def collate(items):
            return to_model_batch(default_collate(items), ssf)

        return collate

    def _sync_rank(self):
        # Upstream seeds shuffling with seed + global_rank; the rank is only known
        # once we're inside the DDP process (after the datamodule is attached).
        if getattr(self, "trainer", None) is not None:
            self._dm.global_rank = int(getattr(self.trainer, "global_rank", 0))

    def _make_loader(self, stage: str):
        import importlib
        from torch.utils.data import DataLoader, IterableDataset

        get_dataset = importlib.import_module("src.dataset").get_dataset
        updm = importlib.import_module("src.dataset.data_module")

        dm = self._dm
        loader_cfg = getattr(dm.data_loader_cfg, stage)
        dataset = get_dataset(dm.dataset_cfg, stage, dm.step_tracker)
        dataset = dm.dataset_shim(dataset, stage)

        if stage == "val":  # upstream limits validation to one batch per scene
            dataset = updm.ValidationWrapper(dataset, 1)

        is_iter = isinstance(dataset, IterableDataset)

        # Train: seed the loader generator from the epoch so the shuffle stream
        # advances across epochs and is reproducible. In exact-resume mode the
        # seed is a function of the epoch ONLY (no resumed_step term) so the
        # interrupted epoch's order is identical on resume and the stateful loader
        # can fast-forward to the exact same samples; otherwise the resumed step is
        # folded in so the stream advances to a fresh shuffle rather than replaying.
        # Val/test keep the upstream deterministic generator.
        if stage == "train" and loader_cfg.seed is not None:
            import torch as _torch

            epoch = int(getattr(getattr(self, "trainer", None), "current_epoch", 0))
            seed_resumed_step = 0 if self._resumable_loader else self._resumed_step
            gen = _torch.Generator()
            gen.manual_seed(
                self._train_seed(loader_cfg.seed, int(dm.global_rank), epoch, seed_resumed_step)
            )
        else:
            gen = dm.get_generator(loader_cfg)

        persistent = dm.get_persistent(loader_cfg)
        if stage == "train" and self._resumable_loader:
            # Workers must respawn each epoch so the per-epoch seed is re-applied;
            # with persistent workers the RNG would drift and break reproducibility.
            persistent = False

        kwargs = dict(
            batch_size=loader_cfg.batch_size,
            num_workers=loader_cfg.num_workers,
            generator=gen,
            worker_init_fn=updm.worker_init_fn,
            persistent_workers=persistent,
            pin_memory=self._pin_memory,
            collate_fn=self._collate_fn(),
        )
        if stage == "train":
            kwargs["shuffle"] = not is_iter
        elif stage == "test":
            kwargs["shuffle"] = False
        if loader_cfg.num_workers > 0 and self._prefetch_factor is not None:
            kwargs["prefetch_factor"] = self._prefetch_factor

        loader_cls = (
            ResumableDataLoader if (stage == "train" and self._resumable_loader) else DataLoader
        )
        return loader_cls(dataset, **kwargs)

    def train_dataloader(self):
        self._sync_rank()
        return self._make_loader("train")

    def val_dataloader(self):
        self._sync_rank()
        return self._make_loader("val")

    def test_dataloader(self):
        self._sync_rank()
        return self._make_loader("test")


class StepTrackerCallback(pl.Callback):
    """Push the trainer's global step into the upstream StepTracker."""

    @staticmethod
    def _update(trainer) -> None:
        dm = trainer.datamodule
        tracker = getattr(dm, "step_tracker", None)
        if tracker is not None and hasattr(tracker, "set_step"):
            tracker.set_step(int(trainer.global_step))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._update(trainer)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:
        self._update(trainer)


def _import_upstream(repo_root: str, repo_name: str):
    """Import the upstream dataset package objects from a repo checkout."""
    add_repo_to_path(repo_root, repo_name=repo_name)
    dm_mod = importlib.import_module("src.dataset.data_module")
    ds_mod = importlib.import_module("src.dataset")
    st_mod = importlib.import_module("src.misc.step_tracker")
    return dm_mod, ds_mod, st_mod


def _view_sampler_cfg(VSCfg, *, num_context_views, num_target_views, knobs):
    return VSCfg(
        name="boundedv2",
        num_context_views=num_context_views,
        num_target_views=num_target_views,
        min_distance_between_context_views=knobs["min_distance_between_context_views"],
        max_distance_between_context_views=knobs["max_distance_between_context_views"],
        max_distance_to_context_views=knobs["max_distance_to_context_views"],
        context_gap_warm_up_steps=knobs["context_gap_warm_up_steps"],
        target_gap_warm_up_steps=knobs["target_gap_warm_up_steps"],
        initial_min_distance_between_context_views=knobs["initial_min_distance_between_context_views"],
        initial_max_distance_between_context_views=knobs["initial_max_distance_between_context_views"],
        initial_max_distance_to_context_views=knobs["initial_max_distance_to_context_views"],
        extra_views_sampling_strategy=knobs["extra_views_sampling_strategy"],
        target_views_replace_sample=knobs["target_views_replace_sample"],
    )


def _loader_cfg(DataLoaderCfg, DataLoaderStageCfg, *, batch_size, num_workers, seed, persistent_workers, test_batch_size=1):
    def stage(bs):
        return DataLoaderStageCfg(
            batch_size=bs,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            seed=seed,
        )
    # test_step evaluates one scene per step (per-scene metrics and image/video
    # dumps), so the test loader is batch_size 1 regardless of the training size.
    return DataLoaderCfg(train=stage(batch_size), test=stage(test_batch_size), val=stage(batch_size))


def build_re10k(
    *,
    mvsplat_root: str,
    dataset_roots: Sequence[str],
    image_shape: Sequence[int],
    num_context_views: int,
    num_target_views: int,
    view_sampler_knobs: dict,
    eval_index_path: str | None = None,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
    persistent_workers: bool = True,
    near: float = 1.0,
    far: float = 100.0,
    make_baseline_1: bool = False,
    baseline_scale_bounds: bool = False,
    augment: bool = True,
    max_fov: float = 100.0,
    baseline_epsilon: float = 1e-3,
    scene_scale_factor: float = 1.0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    resumable_loader: bool = False,
    global_rank: int = 0,
):
    dm_mod, ds_mod, st_mod = _import_upstream(mvsplat_root, "ZPressor/MVSplat")
    DatasetRE10kCfg = ds_mod.DatasetRE10kCfg
    VSCfg = importlib.import_module(
        "src.dataset.view_sampler.view_sampler_bounded_v2"
    ).ViewSamplerBoundedV2Cfg

    # Deterministic eval uses the upstream "evaluation" view sampler, which reads
    # fixed per-scene {context, target} indices from a JSON index (e.g. the
    # c3g_re10k_ctx_*_trg_3v.json files under assets/eval_index/re10k). When no
    # index is given we keep the random bounded_v2 sampler used for training.
    if eval_index_path is not None:
        EvalVSCfg = importlib.import_module(
            "src.dataset.view_sampler.view_sampler_evaluation"
        ).ViewSamplerEvaluationCfg
        # Resolve a relative index path robustly: prefer cwd (back-compat), then
        # fall back to the repo root so the shipped assets/eval_index/... configs
        # work regardless of the launch directory.
        idx_path = Path(eval_index_path).expanduser()
        if not idx_path.is_absolute() and not idx_path.exists():
            repo_root = Path(__file__).resolve().parents[2]
            if (repo_root / idx_path).exists():
                idx_path = repo_root / idx_path
        view_sampler = EvalVSCfg(
            name="evaluation",
            index_path=idx_path.resolve(),
            num_context_views=num_context_views,
        )
    else:
        view_sampler = _view_sampler_cfg(
            VSCfg, num_context_views=num_context_views,
            num_target_views=num_target_views, knobs=view_sampler_knobs,
        )

    dataset_cfg = DatasetRE10kCfg(
        name="re10k",
        roots=[Path(p).expanduser().resolve() for p in dataset_roots],
        image_shape=list(image_shape),
        background_color=[0.0, 0.0, 0.0],
        cameras_are_circular=False,
        overfit_to_scene=None,
        view_sampler=view_sampler,
        baseline_epsilon=baseline_epsilon,
        max_fov=max_fov,
        make_baseline_1=make_baseline_1,
        augment=augment,
        test_len=-1,
        test_chunk_interval=1,
        test_times_per_scene=1,
        skip_bad_shape=True,
        near=near,
        far=far,
        baseline_scale_bounds=baseline_scale_bounds,
        shuffle_val=True,
    )
    step_tracker = st_mod.StepTracker()
    loader_cfg = _loader_cfg(
        dm_mod.DataLoaderCfg, dm_mod.DataLoaderStageCfg,
        batch_size=batch_size, num_workers=num_workers, seed=seed,
        persistent_workers=persistent_workers,
    )
    upstream = dm_mod.DataModule(dataset_cfg, loader_cfg, step_tracker, global_rank=global_rank)
    return UpstreamBackedDataModule(
        upstream, step_tracker, scene_scale_factor,
        pin_memory=pin_memory, prefetch_factor=prefetch_factor,
        resumable_loader=resumable_loader,
    )


def build_dl3dv(
    *,
    depthsplat_root: str,
    dataset_roots: Sequence[str],
    image_shape: Sequence[int],
    packed_image_shape: Sequence[int],
    num_context_views: int,
    num_target_views: int,
    view_sampler_knobs: dict,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
    persistent_workers: bool = True,
    near: float = 1.0,
    far: float = 100.0,
    make_baseline_1: bool = False,
    baseline_scale_bounds: bool = False,
    augment: bool = True,
    max_fov: float = 100.0,
    baseline_epsilon: float = 1e-3,
    scene_scale_factor: float = 1.0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    resumable_loader: bool = False,
    global_rank: int = 0,
):
    dm_mod, ds_mod, st_mod = _import_upstream(depthsplat_root, "DepthSplat")
    DatasetDL3DVCfg = ds_mod.DatasetDL3DVCfg
    VSCfg = importlib.import_module(
        "src.dataset.view_sampler.view_sampler_bounded_v2"
    ).ViewSamplerBoundedV2Cfg

    dataset_cfg = DatasetDL3DVCfg(
        name="dl3dv",
        roots=[Path(p).expanduser().resolve() for p in dataset_roots],
        image_shape=list(image_shape),
        background_color=[0.0, 0.0, 0.0],
        cameras_are_circular=False,
        overfit_to_scene=None,
        view_sampler=_view_sampler_cfg(
            VSCfg, num_context_views=num_context_views,
            num_target_views=num_target_views, knobs=view_sampler_knobs,
        ),
        baseline_epsilon=baseline_epsilon,
        max_fov=max_fov,
        make_baseline_1=make_baseline_1,
        augment=augment,
        test_len=-1,
        test_chunk_interval=1,
        train_times_per_scene=1,
        test_times_per_scene=1,
        ori_image_shape=list(packed_image_shape),
        view_group=[num_context_views + num_target_views],
        skip_bad_shape=True,
        near=near,
        far=far,
        baseline_scale_bounds=baseline_scale_bounds,
        shuffle_val=True,
    )
    step_tracker = st_mod.StepTracker()
    loader_cfg = _loader_cfg(
        dm_mod.DataLoaderCfg, dm_mod.DataLoaderStageCfg,
        batch_size=batch_size, num_workers=num_workers, seed=seed,
        persistent_workers=persistent_workers,
    )
    upstream = dm_mod.DataModule(dataset_cfg, loader_cfg, step_tracker, global_rank=global_rank)
    return UpstreamBackedDataModule(
        upstream, step_tracker, scene_scale_factor,
        pin_memory=pin_memory, prefetch_factor=prefetch_factor,
        resumable_loader=resumable_loader,
    )

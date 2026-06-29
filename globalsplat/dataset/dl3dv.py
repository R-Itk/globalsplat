"""DL3DV (DepthSplat) data module: thin wrapper over the upstream DataModule.

All loading/sampling/augmentation/collation is upstream code; we only add scene
normalization + pixel intrinsics + the batch rename (see ``preprocessing.py``).
"""
from __future__ import annotations

from typing import Sequence

from .data_module import build_dl3dv, StepTrackerCallback


class DepthSplatStepTrackerCallback(StepTrackerCallback):
    pass


def DepthSplatDL3DVDataModule(
    *,
    depthsplat_root: str,
    dataset_roots: Sequence[str],
    image_shape: Sequence[int] = (256, 448),
    packed_image_shape: Sequence[int] = (270, 480),
    num_context_views: int = 24,
    num_target_views: int = 12,
    min_distance_between_context_views: int = 20,
    max_distance_between_context_views: int = 50,
    initial_min_distance_between_context_views: int = 20,
    initial_max_distance_between_context_views: int = 50,
    context_gap_warm_up_steps: int = 0,
    max_distance_to_context_views: int = 0,
    target_gap_warm_up_steps: int = 0,
    initial_max_distance_to_context_views: int = 0,
    extra_views_sampling_strategy: str = "farthest_point",
    target_views_replace_sample: bool = False,
    batch_size: int = 1,
    num_workers: int = 8,
    seed: int = 42,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    near: float = 1.0,
    far: float = 100.0,
    make_baseline_1: bool = False,
    baseline_scale_bounds: bool = False,
    augment: bool = True,
    scene_scale_factor: float = 1.0,
    resumable_loader: bool = False,
    **_ignored,
):
    knobs = dict(
        min_distance_between_context_views=min_distance_between_context_views,
        max_distance_between_context_views=max_distance_between_context_views,
        initial_min_distance_between_context_views=initial_min_distance_between_context_views,
        initial_max_distance_between_context_views=initial_max_distance_between_context_views,
        context_gap_warm_up_steps=context_gap_warm_up_steps,
        max_distance_to_context_views=max_distance_to_context_views,
        target_gap_warm_up_steps=target_gap_warm_up_steps,
        initial_max_distance_to_context_views=initial_max_distance_to_context_views,
        extra_views_sampling_strategy=extra_views_sampling_strategy,
        target_views_replace_sample=target_views_replace_sample,
    )
    return build_dl3dv(
        depthsplat_root=depthsplat_root,
        dataset_roots=dataset_roots,
        image_shape=image_shape,
        packed_image_shape=packed_image_shape,
        num_context_views=num_context_views,
        num_target_views=num_target_views,
        view_sampler_knobs=knobs,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        persistent_workers=persistent_workers,
        near=near,
        far=far,
        make_baseline_1=make_baseline_1,
        baseline_scale_bounds=baseline_scale_bounds,
        augment=augment,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        scene_scale_factor=scene_scale_factor,
        resumable_loader=resumable_loader,
    )

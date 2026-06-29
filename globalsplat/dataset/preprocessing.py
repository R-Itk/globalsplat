"""The only additions on top of the upstream (ZPressor/MVSplat/DepthSplat) data
pipeline: scene normalization, intrinsics pre-processing, and a thin rename of
the upstream ``{context, target}`` batch into the model's ``{inputs, targets}``.

Everything else (chunk loading, view sampling, augmentation, cropping, the
DataLoader, seeding, and collation) is upstream code, used unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Intrinsics pre-processing
# -----------------------------------------------------------------------------

def to_pixel_intrinsics(K: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert upstream normalized intrinsics ([0,1] image fractions) to pixels.

    The upstream RE10K/DL3DV loaders (and their crop shim) keep intrinsics
    normalized. The renderer and ray construction need pixel-space K. The check
    keeps the op idempotent if pixel intrinsics are ever passed in.
    """
    K = K.clone()
    fx_med = float(K[..., 0, 0].detach().float().median().cpu().item())
    fy_med = float(K[..., 1, 1].detach().float().median().cpu().item())
    if fx_med < 10.0 and fy_med < 10.0:
        K[..., 0, :] *= float(W)
        K[..., 1, :] *= float(H)
    return K


# -----------------------------------------------------------------------------
# Scene normalization (canonical average-camera frame + constellation scale)
# Matches the paper (Sec 3.2, YoNoSplat-style): align cameras to the average
# camera frame, then scale translations by the camera-constellation diameter.
# -----------------------------------------------------------------------------

def _normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)


def _c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    R = c2w[..., :3, :3]
    t = c2w[..., :3, 3:4]
    R_inv = np.swapaxes(R, -1, -2)
    t_inv = -R_inv @ t
    w2c = np.zeros_like(c2w)
    w2c[..., :3, :3] = R_inv
    w2c[..., :3, 3] = t_inv[..., 0]
    w2c[..., 3, 3] = 1.0
    return w2c


def scene_normalize_c2w(
    c2w_union: torch.Tensor,
    scene_scale_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Normalize one scene's union of context+target camera-to-world matrices.

    Args:
        c2w_union: ``[V, 4, 4]`` camera-to-world matrices (context then target).
        scene_scale_factor: multiplier on the constellation diameter.

    Returns:
        ``(norm_c2w[V,4,4], norm_w2c[V,4,4], scene_scale, metric_scale)``.
    """
    if c2w_union.ndim != 3 or c2w_union.shape[-2:] != (4, 4):
        raise ValueError(f"Expected c2w_union [V,4,4], got {tuple(c2w_union.shape)}.")

    device, dtype = c2w_union.device, c2w_union.dtype
    c2w = c2w_union.detach().cpu().numpy().astype(np.float32)

    # Average-camera canonical frame.
    center = c2w[:, :3, 3].mean(axis=0)
    avg_forward = _normalize_vector(c2w[:, :3, 2].mean(axis=0))
    avg_down = c2w[:, :3, 1].mean(axis=0)
    avg_right = _normalize_vector(np.cross(avg_down, avg_forward))
    avg_down = _normalize_vector(np.cross(avg_forward, avg_right))

    avg_pose = np.eye(4, dtype=np.float32)
    avg_pose[:3, :3] = np.stack([avg_right, avg_down, avg_forward], axis=-1)
    avg_pose[:3, 3] = center
    world_to_avgcam = np.linalg.inv(avg_pose).astype(np.float32)
    c2w = world_to_avgcam @ c2w

    # Scale by the camera-constellation diameter.
    cam_centers = c2w[:, :3, 3]
    diffs = cam_centers[:, None, :] - cam_centers[None, :, :]
    camera_scale = float(np.sqrt(np.max(np.sum(diffs * diffs, axis=-1))))
    scene_scale = float(scene_scale_factor) * camera_scale
    if not np.isfinite(scene_scale) or scene_scale <= 1e-12:
        scene_scale = 1.0
    c2w[:, :3, 3] /= scene_scale

    norm_c2w = torch.from_numpy(c2w).to(device=device, dtype=dtype)
    norm_w2c = torch.from_numpy(_c2w_to_w2c(c2w)).to(device=device, dtype=dtype)
    metric_scale = 10.0  # RE10K/ACID have no metric depth; kept for the frustum bound
    return norm_c2w, norm_w2c, scene_scale, metric_scale


# -----------------------------------------------------------------------------
# Upstream batch -> model batch
# -----------------------------------------------------------------------------

def _first(d: Dict[str, Any], names) -> Any:
    for n in names:
        if n in d:
            return d[n]
    raise KeyError(f"none of {tuple(names)} in {sorted(d.keys())}")


def to_model_batch(batch: Dict[str, Any], scene_scale_factor: float = 1.0) -> Dict[str, Any]:
    """Convert an upstream collated ``{context, target, scene}`` batch into the
    model's ``{inputs, targets, scene_info}`` format.

    Upstream extrinsics are camera-to-world (``w2c.inverse()``); intrinsics are
    normalized. We only (1) normalize the scene, (2) make intrinsics pixel-space,
    and (3) rename keys / add ``c2w`` and ``frame_ids``.
    """
    ctx, tgt = batch["context"], batch["target"]

    ctx_img = _first(ctx, ["image", "images"])
    tgt_img = _first(tgt, ["image", "images"])
    ctx_K = _first(ctx, ["intrinsics", "intrinsic"])
    tgt_K = _first(tgt, ["intrinsics", "intrinsic"])
    ctx_c2w = _first(ctx, ["extrinsics", "c2w"])
    tgt_c2w = _first(tgt, ["extrinsics", "c2w"])
    ctx_idx = _first(ctx, ["index", "indices", "frame_ids"])
    tgt_idx = _first(tgt, ["index", "indices", "frame_ids"])

    B, Vc = ctx_img.shape[:2]
    H, W = tgt_img.shape[-2:]

    ctx_K = to_pixel_intrinsics(ctx_K, H, W)
    tgt_K = to_pixel_intrinsics(tgt_K, H, W)

    c2w_union = torch.cat([ctx_c2w, tgt_c2w], dim=1)  # [B, Vc+Vt, 4, 4]

    norm_c2w, norm_w2c, metric, scene = [], [], [], []
    for b in range(B):
        nc, nw, ss, ms = scene_normalize_c2w(c2w_union[b], scene_scale_factor)
        norm_c2w.append(nc); norm_w2c.append(nw)
        scene.append(ss); metric.append(ms)
    norm_c2w = torch.stack(norm_c2w, 0)
    norm_w2c = torch.stack(norm_w2c, 0)
    device, dtype = ctx_img.device, ctx_img.dtype
    scene_scale = torch.tensor(scene, device=device, dtype=dtype)
    metric_scale = torch.tensor(metric, device=device, dtype=dtype)

    out: Dict[str, Any] = {
        "scene_info": {"metric_scale": metric_scale, "scene_scale": scene_scale},
        "inputs": {
            "images": ctx_img,
            "intrinsic": ctx_K,
            "extrinsic": norm_w2c[:, :Vc],
            "c2w": norm_c2w[:, :Vc],
            "frame_ids": ctx_idx,
        },
        "targets": {
            "images": tgt_img,
            "intrinsic": tgt_K,
            "extrinsic": norm_w2c[:, Vc:],
            "c2w": norm_c2w[:, Vc:],
            "frame_ids": tgt_idx,
        },
    }
    if "scene" in batch:
        out["scene_info"]["scene"] = batch["scene"]
    return out

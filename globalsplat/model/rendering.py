from __future__ import annotations

import math

import torch
from ..misc.geometry import matrix_to_quaternion, rotation_6d_to_matrix

from .types import Gaussians


RGB_OUTPUT = "RGB"
RGB_DEPTH_OUTPUT = "RGB+D"
RGB_EXPECTED_DEPTH_OUTPUT = "RGB+ED"


# -----------------------------------------------------------------------------
# Rotation / color helpers
# -----------------------------------------------------------------------------

def _rot_to_quat(rot: torch.Tensor) -> torch.Tensor:
    """Convert rotations to quaternions [B, P, 4].

    Accepts [B, P, 6] (6D rotation) or [B, P, 4] (already a quaternion).
    """
    if rot.shape[-1] == 4:
        return rot.contiguous()
    if rot.shape[-1] != 6:
        raise ValueError(f"ROTATION must be [B,P,6] or [B,P,4], got {tuple(rot.shape)}")

    B, P, _ = rot.shape
    mats = rotation_6d_to_matrix(rot.reshape(-1, 6).contiguous()).contiguous()
    return matrix_to_quaternion(mats).reshape(B, P, 4).contiguous()


def _infer_sh_degree_from_K(K: int) -> int:
    """Infer SH degree from coefficient count K, where K = (degree + 1)^2."""
    degree = int(round(math.sqrt(K) - 1))
    if (degree + 1) * (degree + 1) != K:
        raise ValueError(f"Invalid SH coeff count K={K}; expected a perfect square.")
    return degree


# -----------------------------------------------------------------------------
# Main renderer
# -----------------------------------------------------------------------------

def render_static_batched(
    gaussians: Gaussians,
    meta: dict,
    render_depth: bool = True,
    eps2d_fg: float = 0.05,
    packed_fg: bool = True,
) -> dict:
    """Rasterize the Gaussian set with gsplat and return the flattened layout.

    Args:
        gaussians: predicted ``Gaussians`` (means/scales/rotations/sh/opacities).
        meta: the *target* batch; supplies cameras and image size via the keys
            "images" [B,T,3,H,W], "intrinsic" [B,T,3,3], "extrinsic" (w2c) [B,T,4,4].

    Returns a dict with:
        out["img"]   : [B*T, 3, H, W]
        out["acc"]   : [B*T, 1, H, W, 1]
        out["depth"] : [B*T, H, W, 1]  (only if render_depth=True)
        out["fg_img"], out["fg_acc"], out["fg_depth"] alias the above.
    """
    images = meta.get("images", None)
    if images is None:
        raise KeyError('meta must contain "images" [B,T,3,H,W] to infer H and W')

    # gsplat is imported lazily so the rest of the package (dataset, encoder,
    # converter, tests) can be imported and run without a CUDA/gsplat build.
    from gsplat import rasterization

    B_meta, T, _, H, W = images.shape

    Ks = meta.get("intrinsic", None)
    w2cs = meta.get("extrinsic", None)
    if Ks is None or w2cs is None:
        raise KeyError('meta must contain "intrinsic" and "extrinsic"')

    device = gaussians.means.device
    # gsplat rasterization requires fp32. The encoder may run in bf16/fp16 (so that
    # SDPA uses the FlashAttention kernel), so we always render in fp32 and disable
    # autocast around the rasterizer.
    dtype = torch.float32

    if not torch.is_tensor(Ks):
        Ks = torch.as_tensor(Ks)
    if not torch.is_tensor(w2cs):
        w2cs = torch.as_tensor(w2cs)

    Ks = Ks.to(device=device, dtype=dtype).contiguous()
    w2cs = w2cs.to(device=device, dtype=dtype).contiguous()

    means_fg = gaussians.means.to(device=device, dtype=dtype).contiguous()
    scales_fg = gaussians.scales.to(device=device, dtype=dtype).contiguous()
    quats_fg = _rot_to_quat(gaussians.rotations.to(device=device, dtype=dtype).contiguous())

    op_fg = gaussians.opacities.to(device=device, dtype=dtype).contiguous()
    if op_fg.dim() == 3 and op_fg.shape[-1] == 1:
        op_fg = op_fg.squeeze(-1).contiguous()

    colors_fg = gaussians.sh.to(device=device, dtype=dtype).contiguous()
    sh_degree_fg = _infer_sh_degree_from_K(int(colors_fg.shape[-2]))

    B, P, _ = means_fg.shape
    assert B == B_meta, f"gaussians B={B} but meta B={B_meta}"
    assert Ks.shape[:2] == (B, T) and w2cs.shape[:2] == (B, T)

    render_mode = RGB_EXPECTED_DEPTH_OUTPUT if render_depth else RGB_OUTPUT

    # gsplat needs fp32; disable any outer (bf16/fp16) autocast here.
    with torch.autocast(device_type=device.type, enabled=False):
        fg_colors, fg_alphas, _ = rasterization(
            means=means_fg,
            quats=quats_fg,
            scales=scales_fg,
            opacities=op_fg,
            colors=colors_fg,
            viewmats=w2cs,
            Ks=Ks,
            width=W,
            height=H,
            packed=packed_fg,
            render_mode=render_mode,
            sh_degree=sh_degree_fg,
            backgrounds=None,
            camera_model="pinhole",
            eps2d=eps2d_fg,
            near_plane=1e-2,
        )

    fg_rgb = fg_colors[..., :3]

    BT = B * T
    out = {
        "img": fg_rgb.permute(0, 1, 4, 2, 3).reshape(BT, 3, H, W).contiguous(),
        "acc": fg_alphas.reshape(BT, 1, H, W, 1).contiguous(),
    }
    if render_depth:
        out["depth"] = fg_colors[..., -1:].reshape(BT, H, W, 1).contiguous()

    # Foreground == final image (no background path); alias to avoid recomputing.
    out["fg_img"] = out["img"]
    out["fg_acc"] = out["acc"]
    if render_depth:
        out["fg_depth"] = out["depth"]

    return out

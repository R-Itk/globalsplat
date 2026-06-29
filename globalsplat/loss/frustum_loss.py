"""Differentiable soft frustum regularizer.

Penalizes predicted Gaussian centers that fall outside every input camera
frustum (or behind/too far), with a smooth, bounded penalty. Moved verbatim
from the original ``loss.py`` except that ``max_depth`` is now normalized to a
per-batch tensor at the top of the function (it was indexed as ``max_depth[:, None, None]``
but defaulted to a Python float, which crashed the no-scene-info code path).
"""
import torch
import torch.nn.functional as F


def frustum_soft_loss_w2c(
    means: torch.Tensor,        # (B, N, 3) world coords
    intrinsics: torch.Tensor,   # (B, V, 3, 3)
    extrinsics: torch.Tensor,   # (B, V, 3, 4) or (B, V, 4, 4), w2c
    H: int,
    W: int,
    eps: float = 1e-6,
    tau: float = 10.0,          # scale for soft nonlinearity
    max_depth=100.0,            # float or per-batch tensor of shape (B,)
):
    """
    Differentiable frustum loss:
      - points inside at least one frustum get 0 penalty
      - points outside get a smooth, bounded penalty

    Returns:
        loss: scalar (differentiable w.r.t. means)
    """
    B, N, _ = means.shape
    B2, V, E1, E2 = extrinsics.shape
    assert B == B2

    # Normalize max_depth to a (B,) tensor so the far-plane term below
    # (max_depth[:, None, None]) works whether a float or a tensor is passed.
    if not torch.is_tensor(max_depth):
        max_depth = means.new_full((B,), float(max_depth))
    else:
        max_depth = max_depth.to(device=means.device, dtype=means.dtype).reshape(-1)
        if max_depth.numel() == 1:
            max_depth = max_depth.expand(B)

    # --- extract R, t from w2c ---
    if (E1, E2) == (3, 4):
        R = extrinsics[..., :3]      # (B, V, 3, 3)
        t = extrinsics[..., 3]       # (B, V, 3)
    elif (E1, E2) == (4, 4):
        R = extrinsics[..., :3, :3]  # (B, V, 3, 3)
        t = extrinsics[..., :3,  3]  # (B, V, 3)
    else:
        raise ValueError(f"Bad extrinsics shape {extrinsics.shape}")

    # --- world -> cam ---
    X_cam = torch.einsum("b v i j, b n j -> b v n i", R, means) + t.unsqueeze(2)  # (B,V,N,3)
    z = X_cam[..., 2]  # (B, V, N)

    # --- projection ---
    X_cam_mat = X_cam.transpose(-1, -2)          # (B, V, 3, N)
    x_img = torch.matmul(intrinsics, X_cam_mat)  # (B, V, 3, N)
    x_img = x_img.transpose(-1, -2)              # (B, V, N, 3)

    z_safe = x_img[..., 2].clamp_min(eps)
    u = x_img[..., 0] / z_safe   # (B, V, N)
    v = x_img[..., 1] / z_safe   # (B, V, N)

    # --- per-view violation (continuous, >=0) ---
    left   = F.relu(-u)                  # u < 0
    right  = F.relu(u - (W - 1))         # u > W-1
    top    = F.relu(-v)                  # v < 0
    bottom = F.relu(v - (H - 1))         # v > H-1
    behind = F.relu(1e-2-z)                  # z < 0

    far  = F.relu(z - max_depth[:,None,None])            # z > far plane (e.g., 10 units)

    per_view_violation = left + right + top + bottom + behind  + far # (B, V, N)

    # If a point is inside *any* view, its min violation = 0
    per_point_violation, _ = per_view_violation.min(dim=1)     # (B, N)

    # Smooth, bounded penalty – e.g. log(1 + x/tau) or tanh
    soft_penalty = torch.log1p(per_point_violation / tau)      # (B, N)

    loss = soft_penalty.mean()
    return loss



# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).


# the perception loss code is modified from https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/f5216f312cf82d77f8d20454b5eeb3930324630a/models/networks.py#L1478

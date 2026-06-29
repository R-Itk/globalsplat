import torch
import torch.nn.functional as F


# Adapted from PyTorch3D (BSD): https://github.com/facebookresearch/pytorch3d


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    positive_mask = x > 0
    safe_x = torch.where(positive_mask, x, 1.0)
    return torch.where(positive_mask, torch.sqrt(safe_x), 0.0)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form:
    one in which the real part is non-negative.
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions.

    Args:
        matrix: tensor of shape (..., 3, 3)

    Returns:
        quaternions: tensor of shape (..., 4)
        with real part first.
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}")

    batch_dim = matrix.shape[:-2]

    m00, m01, m02, \
        m10, m11, m12, \
        m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
                ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01],
                dim=-1,
            ),
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20],
                dim=-1,
            ),
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21],
                dim=-1,
            ),
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2],
                dim=-1,
            ),
        ],
        dim=-2,
    )

    flr = torch.tensor(
        0.1,
        dtype=q_abs.dtype,
        device=q_abs.device,
    )

    quat_candidates = quat_by_rijk / (
            2.0 * q_abs[..., None].max(flr)
    )

    indices = q_abs.argmax(dim=-1, keepdim=True)
    gather_indices = indices.unsqueeze(-1).expand(
        batch_dim + (1, 4)
    )

    out = torch.gather(
        quat_candidates,
        dim=-2,
        index=gather_indices,
    ).squeeze(-2)

    return standardize_quaternion(out)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation to rotation matrices.

    Args:
        d6: tensor of shape (..., 6)

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]

    b1 = F.normalize(a1, dim=-1)

    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)

    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-2)
"""Write a GlobalSplat ``Gaussians`` set to a standard 3DGS ``.ply``.

INRIA-format binary PLY (degree-3 SH): the format every web splat viewer reads
(SuperSplat, antimatter15/splat, PlayCanvas). Conventions converted from the
model's output: opacity [0,1] -> logit, linear scale -> log, 6D rotation -> a
(w,x,y,z) quaternion, SH -> f_dc (DC band) + f_rest (channel-major).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_NUM_REST = 45  # degree-3 SH: (16 - 1) coeffs * 3 channels


def ply_properties() -> list[str]:
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(_NUM_REST)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    return props


def write_ply(path, means, scales, quats_wxyz, sh, opacities) -> int:
    """Write one Gaussian set to a binary 3DGS .ply. Tensors are [N, ...] on CPU.

    means [N,3], scales [N,3] (linear), quats_wxyz [N,4], sh [N,K,3], opacities [N(,1)].
    Returns the number of Gaussians written.
    """
    path = Path(path)
    N = means.shape[0]
    xyz = means.float().numpy()
    normals = np.zeros((N, 3), np.float32)
    f_dc = sh[:, 0, :].float().numpy()                              # [N, 3] DC band
    rest = sh[:, 1:, :]                                             # [N, 15, 3]
    f_rest = rest.permute(0, 2, 1).reshape(N, -1).float().numpy()   # [N, 45] channel-major
    opa = opacities.reshape(N).clamp(1e-6, 1 - 1e-6)
    opacity = torch.log(opa / (1 - opa)).float().numpy().reshape(N, 1)   # logit
    scale = torch.log(scales.clamp_min(1e-9)).float().numpy()           # log
    rot = quats_wxyz.float().numpy()                                    # (w,x,y,z)

    data = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scale, rot], axis=1).astype(np.float32)
    assert data.shape[1] == len(ply_properties()), (data.shape, len(ply_properties()))

    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % N
    header += "".join(f"property float {p}\n" for p in ply_properties())
    header += "end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())
    return N

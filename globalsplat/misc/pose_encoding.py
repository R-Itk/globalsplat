from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

# -----------------------------------------------------------------------------
# Ray construction
# -----------------------------------------------------------------------------

# Cache the (integer) pixel grid per (h, w, device). It is identical across
# every forward, so rebuilding it each call is wasted work; the per-camera
# intrinsics math below still runs every call on the cached, broadcast grid.
_PIXEL_GRID_CACHE: dict = {}


def _pixel_grid(h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached ``(x_row, y_row)`` of shape ``[1, h*w]`` (float) for a grid."""
    key = (int(h), int(w), str(device))
    cached = _PIXEL_GRID_CACHE.get(key)
    if cached is None:
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        x_row = xx.reshape(1, -1).float()
        y_row = yy.reshape(1, -1).float()
        cached = (x_row, y_row)
        _PIXEL_GRID_CACHE[key] = cached
    return cached


def compute_rays(
    c2w: torch.Tensor,
    K: torch.Tensor,
    h: int,
    w: int,
    device,
) -> torch.Tensor:
    """
    Build Plücker-style ray features from camera matrices and intrinsics.

    Args:
        c2w: [B, V, 4, 4]
        K: [B, V, 3, 3]
        h: target image height
        w: target image width
        device: torch device

    Returns:
        pose_cond: [B, V, 6, h, w]
            Concatenation of:
                - o x d : [B, V, 3, h, w]
                - d_unit: [B, V, 3, h, w]
    """
    b, v = c2w.shape[:2]

    # Ray/Pluecker geometry is precision-sensitive: keep it in fp32 even when the
    # encoder runs under a bf16/fp16 autocast (so FlashAttention is used elsewhere).
    c2w = c2w.reshape(b * v, 4, 4).to(device=device, dtype=torch.float32)
    K = K.reshape(b * v, 3, 3).to(device=device, dtype=torch.float32)

    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    torch._assert((fx > 0).all(), "fx must be > 0")
    torch._assert((fy > 0).all(), "fy must be > 0")
    torch._assert((cx >= -1).all(), "cx < -1")
    torch._assert((cx <= (w + 1)).all(), "cx > w+1")
    torch._assert((cy >= -1).all(), "cy < -1")
    torch._assert((cy <= (h + 1)).all(), "cy > h+1")

    fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1)

    x_row, y_row = _pixel_grid(h, w, device)
    x = x_row.expand(b * v, -1)
    y = y_row.expand(b * v, -1)

    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)

    ray_d = torch.stack([x, y, z], dim=2)
    with torch.autocast(device_type=ray_d.device.type, enabled=False):
        ray_d = torch.bmm(ray_d.float(), c2w[:, :3, :3].transpose(1, 2).float())

    d_norm = torch.norm(ray_d, dim=2, keepdim=True).clamp_min(1e-8)
    ray_d = ray_d / d_norm

    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)

    ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w)
    ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w)

    o_cross_d = torch.cross(ray_o, ray_d, dim=2)
    pose_cond = torch.cat([o_cross_d, ray_d], dim=2)
    return pose_cond


# -----------------------------------------------------------------------------
# Per-view patch conditioning
# -----------------------------------------------------------------------------

class PatchViewPE(nn.Module):
    """
    Per-view conditioning for patch tokens using:

        e_view = proj(concat(MLP(Kfeat), Fourier(o)))
        tokens_out = tokens + e_view

    Assumptions:
        - Uses c2w only.
        - Camera centers are already normalized.
        - tokens are [B, V, P, D].
        - images are [B, V, C, H, W].
    """

    def __init__(
        self,
        token_dim: int,
        k_hidden: int = 64,
        k_out: int = 64,
        fourier_bands: int = 4,
        use_log_focal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.use_log_focal = bool(use_log_focal)
        self.fourier_bands = int(fourier_bands)

        self.k_mlp = nn.Sequential(
            nn.Linear(4, k_hidden),
            nn.GELU(),
            nn.Linear(k_hidden, k_out),
            nn.GELU(),
        )

        o_dim = 3 * (1 + 2 * self.fourier_bands)
        self.proj = nn.Linear(k_out + o_dim, self.token_dim)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _camera_center_from_c2w(c2w: torch.Tensor) -> torch.Tensor:
        if c2w.dim() != 4 or c2w.shape[-2:] != (4, 4):
            raise ValueError(f"c2w must be [B,V,4,4], got {tuple(c2w.shape)}")
        return c2w[..., :3, 3]

    def _fourier_o(self, o: torch.Tensor) -> torch.Tensor:
        """
        Args:
            o: [B, V, 3], assumed roughly |o| < 1

        Returns:
            [B, V, 3 * (1 + 2 * bands)]
        """
        if self.fourier_bands <= 0:
            return o

        freqs = (
            2
            ** torch.arange(
                self.fourier_bands,
                device=o.device,
                dtype=o.dtype,
            )
        ) * torch.pi

        x = o.unsqueeze(-2) * freqs.view(1, 1, -1, 1)
        sincos = torch.cat([torch.sin(x), torch.cos(x)], dim=-2)
        sincos = sincos.reshape(o.shape[0], o.shape[1], -1)
        return torch.cat([o, sincos], dim=-1)

    def _k_feat(self, K: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Convert intrinsics to normalized per-view features.

        Returns:
            [B, V, 4] = [log(fx / W), log(fy / H), cx / W, cy / H]
            or the non-log focal variant.
        """
        if K.dim() != 4 or K.shape[-2:] != (3, 3):
            raise ValueError(f"K must be [B,V,3,3], got {tuple(K.shape)}")

        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]

        fx_n = fx / float(W)
        fy_n = fy / float(H)
        cx_n = cx / float(W)
        cy_n = cy / float(H)

        if self.use_log_focal:
            fx_n = fx_n.clamp_min(1e-8).log()
            fy_n = fy_n.clamp_min(1e-8).log()

        return torch.stack([fx_n, fy_n, cx_n, cy_n], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        c2w: torch.Tensor,
        K: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.dim() != 4:
            raise ValueError(f"tokens must be [B,V,P,D], got {tuple(tokens.shape)}")

        B, V, P, D = tokens.shape
        if D != self.token_dim:
            raise ValueError(
                f"token dim mismatch: got {D}, expected {self.token_dim}"
            )

        if c2w.shape[:2] != (B, V):
            raise ValueError(
                f"c2w must start with [B,V]=[{B},{V}], got {tuple(c2w.shape)}"
            )
        if K.shape != (B, V, 3, 3):
            raise ValueError(
                f"K must be [B,V,3,3]=[{B},{V},3,3], got {tuple(K.shape)}"
            )

        if images.dim() == 5:
            if images.shape[0] != B or images.shape[1] != V:
                raise ValueError(
                    f"images must be [B,V,C,H,W] with B={B},V={V}, got {tuple(images.shape)}"
                )
            H, W = int(images.shape[-2]), int(images.shape[-1])
        else:
            raise ValueError(f"images must be [B,V,C,H,W], got {tuple(images.shape)}")

        o = self._camera_center_from_c2w(c2w)
        o_enc = self._fourier_o(o)

        kfeat = self._k_feat(K, H=H, W=W)
        kemb = self.k_mlp(kfeat)

        view_in = torch.cat([kemb, o_enc], dim=-1)
        e_view = self.proj(view_in)
        e_view = self.drop(e_view)

        return tokens + e_view.unsqueeze(2)



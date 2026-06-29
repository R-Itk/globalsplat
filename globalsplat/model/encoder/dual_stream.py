"""Two-stream (geometry / texture) slot encoder.

A fixed bank of learnable scene-token "slots" is refined over ``rounds`` rounds.
Each round, per stream, does:

    1. cross-attention: the slots (queries) attend the flattened image/ray token
       memory (keys/values),
    2. ``slot_calib_layers_per_round`` self-attention blocks over the slots,

and the two streams then exchange information through a ``PairAdapter``. The geo
and tex streams are independent within a round and only mix in the PairAdapter.

Slot layout (concatenated along the token axis), per batch element:

    [ aux_query_tokens | scene_tokens (the state) | scale_emb | slot_regs ]
      num_aux_queries      cls_size                  1           slot_reg_amount

``aux_query_tokens`` are extra learnable query slots that participate in
attention but whose decoded output is discarded (they only help shape the other
slots). ``scale_emb`` is the single global token consumed by the decoder.
``slot_regs`` are register slots that are *not* updated by cross-attention
(their cross/MLP deltas are masked) and are returned only when requested.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Block


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * float(init_value))  # [D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class PairAdapter(nn.Module):
    """Pairwise exchange between matching geometry and texture slots."""

    def __init__(self, dim: int, init_value: float = 1e-3) -> None:
        super().__init__()
        self.ln_g = nn.LayerNorm(dim)
        self.ln_t = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2 * dim),
        )
        self.ls_g = LayerScale(dim, init_value)
        self.ls_t = LayerScale(dim, init_value)

    def forward(
        self,
        geo: torch.Tensor,
        tex: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        geo_norm = self.ln_g(geo)
        tex_norm = self.ln_t(tex)
        mixed = self.mlp(torch.cat([geo_norm, tex_norm], dim=-1))
        geo_delta, tex_delta = mixed.chunk(2, dim=-1)
        return geo + self.ls_g(geo_delta), tex + self.ls_t(tex_delta)


class _StreamRound(nn.Module):
    """One stream's worth of one round: slot<-memory cross-attention followed by
    a small self-attention tower over the slots.

    This bundles what used to be eleven parallel per-round ``ModuleList``s
    (``slots_ln``/``mem_ln``/``post_ln``/``lat_q``/``mem_k``/``mem_v``/
    ``out_proj``/``attn_ls``/``mlp``/``mlp_ls``/``tower``) into a single module so
    the data flow is local and readable. The computation is unchanged.
    """

    def __init__(
        self,
        dim_latent: int,
        dim_token: int,
        heads: int,
        *,
        qkv_bias: bool,
        mlp_ratio: float,
        attn_init_values: float,
        mlp_init_values: float,
        readout_ln: bool,
        n_self_attn: int,
    ) -> None:
        super().__init__()
        self.D_lat = int(dim_latent)
        self.D_tok = int(dim_token)
        self.heads = int(heads)
        self.dh = self.D_lat // self.heads
        self.readout_ln = bool(readout_ln)

        if self.readout_ln:
            self.slots_ln = nn.LayerNorm(self.D_lat)
            self.mem_ln = nn.LayerNorm(self.D_tok)
            self.post_ln = nn.LayerNorm(self.D_lat)
        else:
            self.slots_ln = self.mem_ln = self.post_ln = None

        self.lat_q = nn.Linear(self.D_lat, self.D_lat, bias=True)
        self.mem_k = nn.Linear(self.D_tok, self.D_lat, bias=True)
        self.mem_v = nn.Linear(self.D_tok, self.D_lat, bias=True)
        self.out_proj = nn.Linear(self.D_lat, self.D_lat, bias=True)
        self.attn_ls = LayerScale(self.D_lat, init_value=float(attn_init_values))

        hidden_dim = int(round(self.D_lat * float(mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(self.D_lat, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, self.D_lat, bias=True),
        )
        self.mlp_ls = LayerScale(self.D_lat, init_value=float(mlp_init_values))

        self.tower = nn.ModuleList(
            [
                Block(
                    dim=self.D_lat,
                    num_heads=self.heads,
                    init_values=1e-2,
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=bool(qkv_bias),
                    rope=None,
                )
                for _ in range(int(n_self_attn))
            ]
        )

    def forward(
        self,
        slots: torch.Tensor,
        mem: torch.Tensor,
        *,
        reg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # --- cross-attention: slots (Q) attend the token memory (K, V) ---
        if self.readout_ln:
            slots_in = self.slots_ln(slots)
            mem_in = self.mem_ln(mem)
        else:
            slots_in, mem_in = slots, mem

        batch_size, num_slots, dim_latent = slots_in.shape
        _, num_memory, _ = mem_in.shape

        q = self.lat_q(slots_in).view(batch_size, num_slots, self.heads, self.dh).transpose(1, 2)
        k = self.mem_k(mem_in).view(batch_size, num_memory, self.heads, self.dh).transpose(1, 2)
        v = self.mem_v(mem_in).view(batch_size, num_memory, self.heads, self.dh).transpose(1, 2)

        # `reg_mask` (built once per forward) zeroes the deltas for register
        # slots so they are frozen w.r.t. cross-attention/MLP.
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, num_slots, dim_latent)
        context = self.out_proj(context)

        attn_delta = self.attn_ls(context)
        if reg_mask is not None:
            attn_delta = attn_delta * reg_mask

        updated = slots + attn_delta

        mlp_input = self.post_ln(updated) if self.readout_ln else updated
        mlp_delta = self.mlp_ls(self.mlp(mlp_input))
        if reg_mask is not None:
            mlp_delta = mlp_delta * reg_mask

        slots = updated + mlp_delta

        # --- self-attention tower over the slots ---
        for block in self.tower:
            slots = block(slots, pos=None)
        return slots


class DualStreamSlotEncoder(nn.Module):
    """Two-stream (geometry/texture) slot encoder: learnable scene-token slots
    cross-attend the flattened image/ray token memory over several rounds, with
    per-round self-attention and a geo<->tex exchange adapter.
    """

    def __init__(
        self,
        dim_latent: int,
        dim_token: int,
        heads: int,
        rounds: int = 4,
        slot_calib_layers_per_round: int = 2,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        readout_ln: bool = True,
        attn_init_values: float = 1e-3,
        mlp_init_values: float = 1e-3,
        num_aux_queries: int = 4,
        mem_reg_amount: int = 4,
        slot_reg_amount: int = 4,
        use_camera_diff_as_input: bool = False,
    ) -> None:
        super().__init__()

        self.D_lat = int(dim_latent)
        self.D_tok = int(dim_token)
        self.heads = int(heads)
        if self.D_lat % self.heads != 0:
            raise ValueError("dim_latent must be divisible by heads")
        self.dh = self.D_lat // self.heads

        self.num_aux_queries = int(num_aux_queries)
        self.mem_reg_amount = int(mem_reg_amount)
        self.slot_reg_amount = int(slot_reg_amount)
        self.use_camera_diff_as_input = bool(use_camera_diff_as_input)

        self.rounds = int(rounds)
        self.slot_calib_layers_per_round = int(slot_calib_layers_per_round)
        self.readout_ln = bool(readout_ln)

        # ----- Learnable slot-bank tokens (prepended/appended to the state) -----
        # Single global token consumed by the decoder.
        self.scale_emb = nn.Parameter(torch.ones(1, 1, self.D_lat))
        # Extra learnable query slots; participate in attention, output discarded.
        self.aux_query_tokens = nn.Parameter(torch.zeros(1, self.num_aux_queries, self.D_lat))
        # Slot-side register tokens (not updated by cross-attention).
        self.slot_regs = (
            nn.Parameter(torch.zeros(1, self.slot_reg_amount, self.D_lat))
            if self.slot_reg_amount > 0
            else None
        )
        # Memory-side learnable register tokens.
        self.mem_regs = (
            nn.Parameter(torch.zeros(1, self.mem_reg_amount, self.D_tok))
            if self.mem_reg_amount > 0
            else None
        )

        # Project the shared slot bank into the two streams.
        self.slot_to_geo = nn.Sequential(
            nn.LayerNorm(self.D_lat),
            nn.Linear(self.D_lat, self.D_lat, bias=True),
        )
        self.slot_to_tex = nn.Sequential(
            nn.LayerNorm(self.D_lat),
            nn.Linear(self.D_lat, self.D_lat, bias=True),
        )
        self.pair_adapters = nn.ModuleList(
            [PairAdapter(self.D_lat, init_value=1e-3) for _ in range(self.rounds)]
        )

        # Per-stream, per-round refinement modules.
        def make_rounds() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    _StreamRound(
                        dim_latent=self.D_lat,
                        dim_token=self.D_tok,
                        heads=self.heads,
                        qkv_bias=qkv_bias,
                        mlp_ratio=mlp_ratio,
                        attn_init_values=attn_init_values,
                        mlp_init_values=mlp_init_values,
                        readout_ln=self.readout_ln,
                        n_self_attn=self.slot_calib_layers_per_round,
                    )
                    for _ in range(self.rounds)
                ]
            )

        self.geo_rounds = make_rounds()
        self.tex_rounds = make_rounds()

        self._init_parameters()

    def _init_parameters(self) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.aux_query_tokens, std=0.02)
            nn.init.trunc_normal_(self.scale_emb, std=0.02)
            if self.slot_regs is not None:
                nn.init.trunc_normal_(self.slot_regs, std=0.02)
            if self.mem_regs is not None:
                nn.init.trunc_normal_(self.mem_regs, std=0.02)

    def _build_slots(self, batch_size: int, state: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        """Create the shared slot bank before splitting to geo/tex streams.

        Returns ``(slots, cls_size, reg_start, reg_end)`` where ``cls_size`` is
        the number of state (scene) tokens and ``[reg_start:reg_end)`` indexes
        the slot registers within ``slots``.
        """
        if state.shape[-1] != self.D_lat:
            raise ValueError(f"state last dim must be D_lat={self.D_lat}, got {state.shape[-1]}")

        cls_size = state.size(1)
        aux_tokens = self.aux_query_tokens.expand(batch_size, -1, -1).contiguous()
        scale_token = self.scale_emb.expand(batch_size, -1, -1).contiguous()

        parts = [aux_tokens, state, scale_token]
        if self.slot_regs is not None:
            parts.append(self.slot_regs.expand(batch_size, -1, -1))

        slots = torch.cat(parts, dim=1)
        reg_end = slots.size(1)
        reg_start = reg_end - self.slot_reg_amount if self.slot_reg_amount > 0 and self.slot_regs is not None else reg_end
        return slots, cls_size, reg_start, reg_end

    def _build_memory(
        self,
        x: torch.Tensor,
        camera_motion_tokens: Optional[torch.Tensor],
        mem_drop_p: float,
    ) -> torch.Tensor:
        batch_size, num_frames, num_tokens, _ = x.shape
        memory_parts = []

        if self.mem_regs is not None:
            memory_parts.append(self.mem_regs.expand(batch_size, -1, -1))
        if camera_motion_tokens is not None and self.use_camera_diff_as_input:
            memory_parts.append(camera_motion_tokens)

        x_mem = x.reshape(batch_size, num_frames * num_tokens, self.D_tok)

        if mem_drop_p > 0.0 and self.training:
            total_tokens = x_mem.size(1)
            keep = int((1.0 - float(mem_drop_p)) * total_tokens)
            keep = max(1, min(total_tokens, keep))
            idx = torch.randint(
                low=0,
                high=total_tokens,
                size=(batch_size, keep),
                device=x_mem.device,
                dtype=torch.long,
            )
            x_mem = x_mem.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.D_tok))

        memory_parts.append(x_mem)
        return torch.cat(memory_parts, dim=1)

    @staticmethod
    def _make_reg_mask(
        num_slots: int,
        reg_start: int,
        reg_end: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Build the [1, num_slots, 1] mask that freezes register slots, once.

        Identical for both streams and every round, so it is computed a single
        time per forward instead of inside each round/stream.
        """
        if reg_end <= reg_start:
            return None
        reg_mask = torch.ones((num_slots,), device=device, dtype=dtype)
        reg_mask[reg_start:reg_end] = 0.0
        return reg_mask.view(1, num_slots, 1)

    def _run_rounds(
        self,
        geo: torch.Tensor,
        tex: torch.Tensor,
        mem: torch.Tensor,
        reg_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for round_idx in range(self.rounds):
            # Streams are independent until the pair adapter, so each is run fully.
            geo = self.geo_rounds[round_idx](geo, mem, reg_mask=reg_mask)
            tex = self.tex_rounds[round_idx](tex, mem, reg_mask=reg_mask)
            geo, tex = self.pair_adapters[round_idx](geo, tex)
        return geo, tex

    def _pack_outputs(
        self,
        geo: torch.Tensor,
        tex: torch.Tensor,
        cls_size: int,
        reg_start: int,
        reg_end: int,
        return_slot_regs: bool,
    ):
        # Auxiliary query slots: averaged across streams, then discarded by callers.
        aux_geo = geo[:, : self.num_aux_queries]
        aux_tex = tex[:, : self.num_aux_queries]
        aux_out = 0.5 * (aux_geo + aux_tex)

        encoded_start = self.num_aux_queries
        encoded_end = encoded_start + cls_size
        encoded_out = (
            tex[:, encoded_start:encoded_end].contiguous(),
            geo[:, encoded_start:encoded_end].contiguous(),
        )

        scale_start = encoded_end
        scale_end = scale_start + 1
        scale_out = 0.5 * (geo[:, scale_start:scale_end] + tex[:, scale_start:scale_end])

        if not return_slot_regs:
            return aux_out.contiguous(), encoded_out, scale_out.contiguous()

        slot_regs_out = None
        if self.slot_regs is not None and self.slot_reg_amount > 0:
            slot_regs_out = (
                tex[:, reg_start:reg_end].contiguous(),
                geo[:, reg_start:reg_end].contiguous(),
            )
        return aux_out.contiguous(), encoded_out, scale_out.contiguous(), slot_regs_out

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        camera_motion_tokens: Optional[torch.Tensor] = None,
        return_slot_regs: bool = False,
        mem_drop_p: float = 0.0,
    ):
        """
        Args:
            x: Input tokens of shape [B, S, N, D_tok].
            state: Latent state tokens of shape [B, L, D_lat]. Required.
            camera_motion_tokens: Optional motion tokens of shape [B, S, D_tok].
            return_slot_regs: Whether to also return slot register outputs.
            mem_drop_p: Drop probability applied to flattened token memory.
        """
        batch_size, num_frames, _, token_dim = x.shape
        if token_dim != self.D_tok:
            raise ValueError(f"x last dim must be D_tok={self.D_tok}, got {token_dim}")

        if camera_motion_tokens is not None:
            expected_shape = (batch_size, num_frames, self.D_tok)
            if camera_motion_tokens.shape != expected_shape:
                raise ValueError(
                    f"camera_motion_tokens must be {expected_shape}, got {tuple(camera_motion_tokens.shape)}"
                )
        elif self.use_camera_diff_as_input:
            raise ValueError("camera_motion_tokens must be provided when use_camera_diff_as_input=True")

        slots0, cls_size, reg_start, reg_end = self._build_slots(batch_size, state)
        geo = self.slot_to_geo(slots0)
        tex = self.slot_to_tex(slots0)
        mem = self._build_memory(x, camera_motion_tokens, mem_drop_p)
        reg_mask = self._make_reg_mask(
            geo.size(1), reg_start, reg_end, device=geo.device, dtype=geo.dtype
        )
        geo, tex = self._run_rounds(geo, tex, mem, reg_mask)
        return self._pack_outputs(geo, tex, cls_size, reg_start, reg_end, return_slot_regs)

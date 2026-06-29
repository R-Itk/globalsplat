"""GlobalSplat LightningModule and the batch helpers used during training.

Moved from the original ``train.py`` (the model definition + training/eval
logic). The subset-consistency objective, staged coarse-to-fine schedule, and
all loss wiring are preserved exactly; only import paths were updated.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .optim import build_optimizer_and_scheduler
from .rendering import render_static_batched
from .types import Gaussians
from ..loss.rendering_loss import LossComputer
from ..loss.frustum_loss import frustum_soft_loss_w2c


@torch.no_grad()
def make_targets_include_inputs_sorted(
    batch: Dict[str, Any],
    *,
    inputs_key: str = "inputs",
    targets_key: str = "targets",
    sort_key: str = "frame_ids",
    view_dim: int = 1,
    keys_to_merge: Optional[Iterable[str]] = None,
    add_is_input: bool = True,
    in_place: bool = True,
) -> Dict[str, Any]:
    if not in_place:
        batch = dict(batch)
        batch[inputs_key] = dict(batch.get(inputs_key, {}))
        batch[targets_key] = dict(batch.get(targets_key, {}))

    inp: Dict[str, torch.Tensor] = batch[inputs_key]
    tgt: Dict[str, torch.Tensor] = batch[targets_key]

    fin = inp[sort_key]
    ftg = tgt[sort_key]

    if keys_to_merge is None:
        keys_to_merge = [k for k in tgt.keys() if k in inp]

    def fast_concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # torch.cat already allocates once and copies both halves; the previous
        # hand-rolled empty()+copy_() did the same thing less readably.
        return torch.cat([a, b], dim=view_dim)

    merged: Dict[str, torch.Tensor] = {}
    for k, v_t in tgt.items():
        if k in inp and k in keys_to_merge:
            merged[k] = fast_concat(inp[k], v_t)
        else:
            merged[k] = v_t

    merged[sort_key] = fast_concat(fin, ftg)

    f_all = merged[sort_key]
    V = int(f_all.shape[view_dim])
    n_in = int(fin.shape[view_dim])

    if add_is_input:
        if f_all.ndim == 1:
            is_input = torch.zeros((V,), device=f_all.device, dtype=torch.bool)
            is_input[:n_in] = True
        else:
            is_input = torch.zeros_like(f_all, dtype=torch.bool)
            sl = [slice(None)] * f_all.ndim
            sl[view_dim] = slice(0, n_in)
            is_input[tuple(sl)] = True

    perm = torch.argsort(f_all, dim=view_dim, stable=True)

    def reorder(v: torch.Tensor) -> torch.Tensor:
        if v.shape[view_dim] != V:
            return v
        if perm.ndim == 1:
            return v.index_select(view_dim, perm)
        idx = perm
        while idx.ndim < v.ndim:
            idx = idx.unsqueeze(-1)
        idx = idx.expand(*v.shape[:view_dim], V, *v.shape[view_dim + 1 :])
        return torch.gather(v, dim=view_dim, index=idx)

    for k, v in list(merged.items()):
        merged[k] = reorder(v)

    if add_is_input:
        merged["is_input"] = reorder(is_input)

    batch[targets_key] = merged
    return batch


@torch.no_grad()
def make_anchor_alternating_input_subsets_and_shared_targets(
    *,
    inputs: Dict[str, Any],
    targets: Dict[str, Any],
    min_shared_targets: int = 1,
    flip: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], torch.Tensor]:
    """
    Build two input subsets A/B from inputs only:
      - sort by input frame_ids
      - duplicate min/max input views into both subsets
      - alternate middle views between A/B (optionally flip parity)
      - pad smaller subset by duplicating its last assigned view so A/B have same #views (for batching)
    Shared targets are the original targets (unchanged).

    Returns:
      inputs_A, inputs_B, shared_targets, valid_batch_mask[B]
    """
    assert "frame_ids" in inputs, "inputs must contain frame_ids"
    frame_ids = inputs["frame_ids"]  # [B,V]
    assert torch.is_tensor(frame_ids) and frame_ids.ndim == 2, "expected inputs['frame_ids'] shape [B,V]"

    B, Vin = frame_ids.shape
    device = frame_ids.device

    # Need at least 2 inputs for min/max anchors and at least min_shared_targets targets
    Tt = int(targets["frame_ids"].shape[1]) if ("frame_ids" in targets and torch.is_tensor(targets["frame_ids"])) else 0
    valid_batch = torch.ones(B, dtype=torch.bool, device=device)
    if Vin < 2 or Tt < int(min_shared_targets):
        valid_batch[:] = False
        return inputs, inputs, targets, valid_batch

    perm = torch.argsort(frame_ids, dim=1, stable=True)  # [B,V]

    # positions in sorted order
    # anchors always included in both
    pos_min = torch.zeros(B, 1, device=device, dtype=torch.long)
    pos_max = torch.full((B, 1), Vin - 1, device=device, dtype=torch.long)

    if Vin == 2:
        posA_sorted = torch.cat([pos_min, pos_max], dim=1)
        posB_sorted = torch.cat([pos_min, pos_max], dim=1)
    else:
        mid = torch.arange(1, Vin - 1, device=device).view(1, -1).expand(B, -1)  # [B, Vin-2]
        # Alternate in sorted space
        if not flip:
            midA = mid[:, 0::2]
            midB = mid[:, 1::2]
        else:
            midA = mid[:, 1::2]
            midB = mid[:, 0::2]

        posA_sorted = torch.cat([pos_min, midA, pos_max], dim=1)
        posB_sorted = torch.cat([pos_min, midB, pos_max], dim=1)

        # pad to equal length (rare if Vin-2 odd)
        LA, LB = posA_sorted.shape[1], posB_sorted.shape[1]
        if LA != LB:
            if LA < LB:
                pad_src = posA_sorted[:, -1:].clone()
                posA_sorted = torch.cat([posA_sorted, pad_src], dim=1)
            else:
                pad_src = posB_sorted[:, -1:].clone()
                posB_sorted = torch.cat([posB_sorted, pad_src], dim=1)

    # map sorted positions back to original input indices
    idxA = torch.gather(perm, 1, posA_sorted)  # [B,VA]
    idxB = torch.gather(perm, 1, posB_sorted)  # [B,VB], same VA after padding

    def _slice_input_views(d: Dict[str, Any], idx: torch.Tensor) -> Dict[str, Any]:
        out = {}
        Vsel = idx.shape[1]
        for k, v in d.items():
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[0] == B and v.shape[1] == Vin:
                g = idx
                while g.ndim < v.ndim:
                    g = g.unsqueeze(-1)
                g = g.expand(B, Vsel, *v.shape[2:])
                out[k] = torch.gather(v, 1, g)
            else:
                out[k] = v
        return out

    inputs_A = _slice_input_views(inputs, idxA)
    inputs_B = _slice_input_views(inputs, idxB)
    shared_targets = targets  # unchanged

    return inputs_A, inputs_B, shared_targets, valid_batch


class GlobalSplatModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_w_dict: Optional[Dict[str, float]] = None,
        batch_size: int = 1,
        eval_mode: bool = False,
        stage_boundaries: Tuple[int, ...] = (10_000, 20_000, 50_000),
        stage_ramp_iters: int = 2_000,
        final_stage: int = 3,
        test_cfg: Optional[Dict[str, Any]] = None,
        upstream_repo_root: Optional[str] = None,
        experiment_name: str = "globalsplat",
        warmup_pct: float = 0.03,
        min_lr_ratio: float = 1.0 / 50.0,
        min_lr_floor: float = 1e-6,
        warmup_epochs_fallback: int = 5,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.eval_mode = eval_mode

        # Warmup -> cosine LR schedule knobs (consumed in configure_optimizers).
        self.warmup_pct = float(warmup_pct)
        self.min_lr_ratio = float(min_lr_ratio)
        self.min_lr_floor = float(min_lr_floor)
        self.warmup_epochs_fallback = int(warmup_epochs_fallback)

        # At eval we only need the learned model weights; the frozen perceptual
        # loss (VGG) is rebuilt from its own weights file and its constants are
        # set at init, so tolerate them being absent/extra in the checkpoint.
        # Real shape mismatches (e.g. wrong latent_rep_token_amount) still error.
        if eval_mode:
            self.strict_loading = False

        # Evaluation (mode=test) configuration, mirroring the upstream TestCfg.
        self.test_cfg = dict(test_cfg or {})
        self.upstream_repo_root = upstream_repo_root
        self.experiment_name = str(experiment_name)

        # Coarse-to-fine capacity curriculum. final_stage selects the eval-time
        # Gaussians-per-token (2**final_stage); boundaries are clipped to it so a
        # lower final_stage simply stops growing capacity earlier.
        #   final_stage 0 -> 1 splat/token   (e.g. GlobalSplat-2K with 2048 latents)
        #   final_stage 3 -> 8 splats/token  (e.g. GlobalSplat-16K / -32K)
        self.final_stage = int(final_stage)
        self.stage_boundaries = tuple(int(b) for b in stage_boundaries)[: self.final_stage]
        self.stage_ramp_iters = int(stage_ramp_iters)

        loss_w_dict = loss_w_dict or {}
        for k, v in loss_w_dict.items():
            setattr(self, k, v)

        self.w_rgb = float(getattr(self, "w_rgb", 1.0))
        self.w_mse = float(getattr(self, "w_mse", 0.5))
        self.w_inview = float(getattr(self, "w_inview", 1e-2))
        # Read in training_step/eval; default here so a loss config that omits it
        # (any config without an explicit w_depth) cannot raise AttributeError.
        self.w_depth = float(getattr(self, "w_depth", 0.0))

        # subset-consistency weights (new)
        self.subset_consistency = bool(getattr(self, "subset_consistency", True))
        self.subset_start_step = int(getattr(self, "subset_start_step", 0))
        self.subset_flip_parity = bool(getattr(self, "subset_flip_parity", True))
        self.subset_min_targets = int(getattr(self, "subset_min_targets", 1))
        self.subset_alpha_consistency_w = float(getattr(self, "subset_alpha_consistency_w", 1e-3))
        self.subset_depth_consistency_w = float(getattr(self, "subset_depth_consistency_w",1e-2))
        # The perceptual-loss network is a frozen, pretrained VGG (~138M params,
        # ~0.5 GB fp32) used only by the training loss. Skip building it at eval
        # so we don't load half a gigabyte of unused weights; on_save/on_load
        # below also keep it out of saved checkpoints (model-only, ~1/3 size).
        if not self.eval_mode:
            self.render_criterion = LossComputer(lpips_w=0.0, l2_w=self.w_mse, perc_w=0.5)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Save only the model's own weights. The perceptual-loss VGG and the
        # (optional) LPIPS network under ``render_criterion.*`` are frozen,
        # pretrained feature extractors re-created identically on load, so they
        # are dropped unconditionally -- persisting them only bloats the file
        # (~0.5 GB of VGG) with weights that are never trained or read back.
        sd = checkpoint.get("state_dict")
        if not sd:
            return
        for key in [k for k in sd if k.startswith("render_criterion.")]:
            del sd[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Re-inject the (identical) perceptual-loss weights that on_save dropped
        # so a strict load during training/resume still succeeds. At eval the
        # module isn't built and strict_loading=False, so any such keys are
        # simply ignored.
        sd = checkpoint.get("state_dict")
        if not sd or not hasattr(self, "render_criterion"):
            return
        for k, v in self.state_dict().items():
            if k.startswith("render_criterion.") and k not in sd:
                sd[k] = v

    @torch.no_grad()
    def _ensure_c2w(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if batch.get("c2w") is None:
            raise KeyError(
                "inputs must contain a 'c2w' camera-to-world tensor; got keys: "
                + ", ".join(sorted(batch.keys()))
            )
        return batch



    def forward(self, inputs: Dict[str, Any]):
        inputs = self._ensure_c2w(inputs)
        return self.model(inputs)

    def _cat_batch_dict(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k in a.keys():
            va, vb = a[k], b[k]
            if torch.is_tensor(va) and torch.is_tensor(vb) and va.ndim >= 1 and vb.ndim >= 1 and va.shape[1:] == vb.shape[1:]:
                out[k] = torch.cat([va, vb], dim=0)
            else:
                out[k] = va
        return out

    def _select_batch_mask(self, d: Dict[str, Any], mask: torch.Tensor) -> Dict[str, Any]:
        out = {}
        B = mask.shape[0]
        for k, v in d.items():
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == B:
                out[k] = v[mask]
            else:
                out[k] = v
        return out

    def _split_preds_batch(self, preds_2b: Gaussians, B: int) -> Tuple[Gaussians, Gaussians]:
        # Gaussians.__getitem__ slices the batch dim; reg is a shared scalar.
        return preds_2b[:B], preds_2b[B:]

    def _split_render_out_batch(self, out_2b: Dict[str, Any], B: int, T: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        BT = B * T
        out_A, out_B = {}, {}
        for k, v in out_2b.items():
            if not torch.is_tensor(v):
                out_A[k] = v
                out_B[k] = v
                continue
            if v.ndim >= 1 and v.shape[0] == 2 * BT:
                out_A[k] = v[:BT]
                out_B[k] = v[BT:]
            else:
                out_A[k] = v
                out_B[k] = v
        return out_A, out_B

    def training_step(self, batch, batch_idx):
        stage, mix = self._stage_schedule(int(self.global_step))
        self.model.set_stage(stage, mix=mix)

        inputs = batch["inputs"]
        trg = batch["targets"]
        scene_info = batch["scene_info"]

        # ---------- subset consistency path (single forward + single render on 2B) ----------
        use_subset = self.subset_consistency and (int(self.global_step) >= self.subset_start_step)
        if use_subset:
            flip = (int(self.global_step) % 2 == 1) if self.subset_flip_parity else False

            inputs_A, inputs_B, shared_tg, valid_batch = make_anchor_alternating_input_subsets_and_shared_targets(
                inputs=inputs,
                targets=trg,
                min_shared_targets=self.subset_min_targets,
                flip=flip,
            )

            if bool(valid_batch.any()):
                inputs_A = self._select_batch_mask(inputs_A, valid_batch)
                inputs_B = self._select_batch_mask(inputs_B, valid_batch)
                shared_tg = self._select_batch_mask(shared_tg, valid_batch)
                scene_info_sub = self._select_batch_mask(scene_info, valid_batch) if scene_info is not None else None

                Bv = shared_tg["images"].shape[0]
                Tv = shared_tg["images"].shape[1]

                inputs_2b = self._cat_batch_dict(inputs_A, inputs_B)
                tg_2b = self._cat_batch_dict(shared_tg, shared_tg)

                preds_2b = self(inputs_2b)
                out_2b = render_static_batched(
                    preds_2b,
                    tg_2b,
                    render_depth=(self.w_depth > 0) or (self.subset_depth_consistency_w > 0),
                )

                preds_A, preds_B = self._split_preds_batch(preds_2b, Bv)
                out_A, out_B = self._split_render_out_batch(out_2b, Bv, Tv)

                # A and B are scored against the SAME shared targets, so the
                # perceptual loss's GT VGG features are identical. Compute them
                # once on the A branch and reuse them for B (saves one full
                # VGG-19 pass over the B*T target images every step).
                loss_A, perc_cache = self.compute_loss_from_out(
                    preds=preds_A, out=out_A, trg=shared_tg, inputs=inputs_A,
                    scene_info=scene_info_sub, return_perc_cache=True,
                )
                loss_B = self.compute_loss_from_out(
                    preds=preds_B, out=out_B, trg=shared_tg, inputs=inputs_B,
                    scene_info=scene_info_sub, perc_cache=perc_cache,
                )

                # -----------------------------
                # Subset consistency (symmetric stop-grad)
                # -----------------------------
                # NOTE:
                #   0.5 * |A - sg(B)| + 0.5 * |B - sg(A)|
                # gives symmetric supervision while blocking same-term co-adaptation.

                subset_alpha_l1 = torch.tensor(0.0, device=out_A["acc"].device, dtype=out_A["acc"].dtype)
                if ("acc" in out_A) and ("acc" in out_B):
                    acc_A = out_A["acc"].float()
                    acc_B = out_B["acc"].float()
                    subset_alpha_l1 = 0.5 * (acc_A - acc_B.detach()).abs().mean() \
                                    + 0.5 * (acc_B - acc_A.detach()).abs().mean()

                subset_depth_l1 = torch.tensor(0.0, device=out_A["depth"].device, dtype=out_A["depth"].dtype)
                if self.subset_depth_consistency_w > 0 and ("depth" in out_A) and ("depth" in out_B):
                    dep_A = out_A["depth"].float()
                    dep_B = out_B["depth"].float()

                    # optional: mask to avoid comparing invalid / empty pixels
                    if ("acc" in out_A) and ("acc" in out_B):
                        m = ((out_A["acc"].float() > 1e-2) & (out_B["acc"].float() > 1e-2)).float()
                        denom = m.sum().clamp_min(1.0)
                        subset_depth_l1 = 0.5 * (((dep_A - dep_B.detach()).abs() * m).sum() / denom) \
                                        + 0.5 * (((dep_B - dep_A.detach()).abs() * m).sum() / denom)
                    else:
                        subset_depth_l1 = 0.5 * (dep_A - dep_B.detach()).abs().mean() \
                                        + 0.5 * (dep_B - dep_A.detach()).abs().mean()

                subset_consistency_loss = (
                    + self.subset_alpha_consistency_w * subset_alpha_l1
                    + self.subset_depth_consistency_w * subset_depth_l1
                )

                total_loss = 0.5 * (loss_A["loss"] + loss_B["loss"]) + subset_consistency_loss

                log_dict = {}
                for k in set(loss_A.keys()).union(loss_B.keys()):
                    if k == "loss":
                        continue
                    va = loss_A.get(k, None)
                    vb = loss_B.get(k, None)
                    if va is None:
                        log_dict[f"train_{k}"] = vb
                    elif vb is None:
                        log_dict[f"train_{k}"] = va
                    else:
                        log_dict[f"train_{k}"] = 0.5 * (va + vb)

                log_dict["train_subset_alpha_l1"] = subset_alpha_l1
                log_dict["train_subset_depth_l1"] = subset_depth_l1
                log_dict["train_subset_consistency"] = subset_consistency_loss
                log_dict["train_loss"] = total_loss

                for k, v in log_dict.items():
                    self.log(k, v, prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)

                if self.global_step % 500 == 0 and self.trainer.is_global_zero:
                    with torch.no_grad():
                        # log the A-branch prediction against the shared targets
                        self._maybe_log_video(preds_A, shared_tg)

                return total_loss

        # ---------- fallback: original single-batch supervised ----------
        preds = self(inputs)
        loss_dict = self.compute_loss(preds, trg, scene_info=scene_info, inputs=inputs)

        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, prog_bar=False, on_step=True, on_epoch=False, sync_dist=False)

        if self.global_step % 500 == 0 and self.trainer.is_global_zero:
            with torch.no_grad():
                batch_new = dict(batch)
                batch_new = make_targets_include_inputs_sorted(batch_new, view_dim=1, in_place=True)
                self._maybe_log_video(preds, batch_new["targets"])

        return loss_dict["loss"]

    def on_validation_start(self):
        stage, mix = self._stage_schedule(int(self.global_step))
        self.model.set_stage(stage, mix=mix)
        if self.trainer.is_global_zero:
            print(f"[val] global_step={int(self.global_step)} stage={stage} mix={mix:.4f}")

    @torch.no_grad()
    def _maybe_log_video(self, preds: Gaussians, trg: Dict[str, Any]) -> None:
        out = render_static_batched(preds, trg, render_depth=True)

        B, T = trg["images"].shape[:2]
        H, W = trg["images"].shape[-2:]
        eps = 1e-6

        def _to_bt1hw(x: torch.Tensor) -> torch.Tensor:
            while x.ndim >= 4 and x.shape[-1] == 1:
                x = x.squeeze(-1)
            if x.ndim == 5:
                if x.shape == (B, T, 1, H, W):
                    return x
                raise ValueError(f"Unexpected 5D shape: {tuple(x.shape)}")
            if x.ndim == 4:
                if x.shape == (B, T, H, W):
                    return x.unsqueeze(2)
                if x.shape == (B * T, 1, H, W):
                    return x.view(B, T, 1, H, W)
                if x.shape == (B * T, H, W, 1):
                    x = x.permute(0, 3, 1, 2)
                    return x.view(B, T, 1, H, W)
                raise ValueError(f"Unexpected 4D shape: {tuple(x.shape)}")
            if x.ndim == 3:
                if x.shape == (B * T, H, W):
                    return x.view(B, T, 1, H, W)
                raise ValueError(f"Unexpected 3D shape: {tuple(x.shape)}")
            raise ValueError(f"Unexpected ndim={x.ndim}, shape={tuple(x.shape)}")

        def _norm_vis(map_bt1hw: torch.Tensor, mask_bt1hw: torch.Tensor) -> torch.Tensor:
            m = mask_bt1hw.to(dtype=map_bt1hw.dtype)
            vmin = map_bt1hw.masked_fill(m == 0, float("inf")).amin(dim=(-2, -1), keepdim=True)
            vmax = map_bt1hw.masked_fill(m == 0, float("-inf")).amax(dim=(-2, -1), keepdim=True)
            bad = (~torch.isfinite(vmin)) | (~torch.isfinite(vmax)) | ((vmax - vmin) < eps)
            if bad.any():
                vmin_fb = map_bt1hw.amin(dim=(-2, -1), keepdim=True)
                vmax_fb = map_bt1hw.amax(dim=(-2, -1), keepdim=True)
                vmin = torch.where(bad, vmin_fb, vmin)
                vmax = torch.where(bad, vmax_fb, vmax)
            vis = (map_bt1hw - vmin) / (vmax - vmin + eps)
            return vis.clamp(0, 1).repeat(1, 1, 3, 1, 1)

        gt_img = trg["images"].float().clamp(0, 1)

        gt_valid = trg.get("valid_mask", None)
        if gt_valid is None:
            gt_valid = torch.ones((B, T, 1, H, W), device=gt_img.device, dtype=torch.bool)
        else:
            if gt_valid.ndim == 4:
                gt_valid = gt_valid.unsqueeze(2)
            gt_valid = gt_valid.bool()

        gt_depth_vis = None
        gt_depth = trg.get("depth", None)
        if isinstance(gt_depth, torch.Tensor):
            try:
                gt_depth_bt1hw = _to_bt1hw(gt_depth.float())
                gt_depth_mask = gt_valid & torch.isfinite(gt_depth_bt1hw) & (gt_depth_bt1hw > 0)
                if bool(gt_depth_mask.any()):
                    gt_depth_vis = _norm_vis(gt_depth_bt1hw, gt_depth_mask)
            except Exception:
                gt_depth_vis = None

        pred_img = out["img"].view(B, T, 3, H, W).clamp(0, 1)

        pred_alpha = out.get("acc", out.get("alpha", None))
        if pred_alpha is None:
            pred_alpha = torch.zeros((B * T, H, W), device=pred_img.device, dtype=pred_img.dtype)
        pred_alpha = _to_bt1hw(pred_alpha).clamp(0, 1)
        pred_alpha_vis = pred_alpha.repeat(1, 1, 3, 1, 1)

        pred_depth = out.get("depth", None)
        if pred_depth is None:
            pred_depth = torch.zeros((B * T, H, W), device=pred_img.device, dtype=pred_img.dtype)
        pred_depth = _to_bt1hw(pred_depth.float())
        pred_depth_mask = (pred_alpha > 1e-2) & torch.isfinite(pred_depth) & (pred_depth > 0)
        pred_depth_vis = _norm_vis(pred_depth, pred_depth_mask)

        if gt_depth_vis is not None:
            top = torch.cat([gt_img[:1], gt_depth_vis[:1], pred_alpha_vis[:1]], dim=-1)
            bot = torch.cat([pred_img[:1], pred_depth_vis[:1], pred_alpha_vis[:1]], dim=-1)
            tag = "Video/GT_vs_Pred__Img_GTDepth_PredDepth_Alpha"
        else:
            top = torch.cat([gt_img[:1], pred_img[:1]], dim=-1)
            bot = torch.cat([pred_depth_vis[:1], pred_alpha_vis[:1]], dim=-1)
            tag = "Video/GT_vs_Pred__Img_PredDepth_Alpha"

        grid = torch.cat([top, bot], dim=-2)
        self.logger.experiment.add_video(
            tag=tag,
            vid_tensor=grid.detach().cpu(),
            global_step=int(self.global_step),
            fps=5,
        )

    def _stage_schedule(self, step: int) -> tuple[int, float]:
        boundaries = self.stage_boundaries
        ramp_iters = self.stage_ramp_iters

        stage = 0
        last_b = 0
        for b in boundaries:
            if step >= b:
                stage += 1
                last_b = b
            else:
                break

        if ramp_iters > 0 and step >= last_b and last_b > 0:
            t = step - last_b
            if t <= 0:
                mix = 0.0
            elif t >= ramp_iters:
                mix = 1.0
            else:
                mix = float(t) / float(ramp_iters)
        else:
            mix = 1.0
        return stage, mix

    def compute_loss(
        self,
        preds: Gaussians,
        trg: Dict[str, Any],
        inputs: Optional[Dict[str, Any]] = None,
        scene_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        out = render_static_batched(preds, trg, render_depth=(self.w_depth > 0))
        return self.compute_loss_from_out(preds=preds, out=out, trg=trg, inputs=inputs, scene_info=scene_info)

    def compute_loss_from_out(
        self,
        *,
        preds: Gaussians,
        out: Dict[str, Any],
        trg: Dict[str, Any],
        inputs: Optional[Dict[str, Any]] = None,
        scene_info: Optional[Dict[str, Any]] = None,
        perc_cache: Any = None,
        return_perc_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, _, H, W = trg["images"].shape
        BT = B * T

        device = preds.means.device
        dtype = preds.means.dtype
        total: Dict[str, torch.Tensor] = {"loss": torch.tensor(0.0, device=device, dtype=dtype)}

        metric_scales = scene_info.get("metric_scale", None) if scene_info is not None else None
        scene_scales = scene_info.get("scene_scale", None) if scene_info is not None else None

        gt_img = trg["images"].flatten(0, 1)

        if inputs is not None:
            Hi, Wi = inputs["images"].shape[-2:]
            if (metric_scales is not None) and (scene_scales is not None):
                max_depth = (metric_scales / scene_scales) * 4.0
            else:
                max_depth = 4.0

            fr_i = frustum_soft_loss_w2c(
                preds.means,
                inputs["intrinsic"],
                inputs["extrinsic"],
                Hi,
                Wi,
                max_depth=max_depth,
            )
            total["inview_reg"] = self.w_inview * fr_i
            total["loss"] = total["loss"] + total["inview_reg"]

        if preds.reg is not None:
            total["other_regs"] = preds.reg
            total["loss"] = total["loss"] + preds.reg

        #render losses
        pred_img = out["img"]
        rgb_losses, perc_cache = self.render_criterion(
            pred_img, gt_img, perc_target_cache=perc_cache, return_perc_target_cache=True
        )
        total.update({f"render_{k}": v for k, v in rgb_losses.items()})
        total["loss"] = total["loss"] + self.w_rgb * rgb_losses["loss"]

        with torch.no_grad():
            eps = 1e-8
            acc_hw = out["acc"].reshape(BT, H, W).clamp(0, 1)
            acc_flat = acc_hw.flatten(1)

            total["alpha_mean"] = acc_flat.mean()
            total["alpha_var"] = acc_flat.var(unbiased=False)

            sc = preds.scales.float()
            sc_size = sc.mean(dim=-1).reshape(-1)
            total["scale_mean"] = sc_size.mean()
            total["scale_var"] = sc_size.var(unbiased=False)

            pred_psnr = out["img"].float().clamp(0, 1)
            gt_psnr = gt_img.float()
            if gt_psnr.max() > 1.5:
                gt_psnr = gt_psnr / 255.0
            gt_psnr = gt_psnr.clamp(0, 1)
            total["psnr"] = 10.0 * torch.log10(1.0 / ((pred_psnr - gt_psnr).pow(2).mean() + eps))

        if return_perc_cache:
            return total, perc_cache
        return total

    def on_train_epoch_start(self):
        dl = self.trainer.train_dataloader
        bs = getattr(dl, "batch_sampler", None)
        if bs is not None and hasattr(bs, "set_epoch"):
            bs.set_epoch(self.current_epoch)

    @torch.no_grad()
    def eval_psnr(self, batch, *, log_video_n: int = 0, fps: int = 4, tag: str = "val"):
        inputs = batch["inputs"]
        trg = batch["targets"]

        preds = self.model(inputs)
        out = render_static_batched(preds, trg, render_depth=False)

        pred = out["img"].float()
        gt = trg["images"].flatten(0, 1).float()

        if pred.numel() and pred.max().detach().item() > 1.5:
            pred = pred / 255.0
        if gt.numel() and gt.max().detach().item() > 1.5:
            gt = gt / 255.0
        pred = pred.clamp(0, 1)
        gt = gt.clamp(0, 1)

        mse = F.mse_loss(pred, gt, reduction="none").mean(dim=(1, 2, 3)).clamp_min(1e-8)
        psnr = (10.0 * torch.log10(1.0 / mse)).mean()
        self.log(f"{tag}_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if log_video_n and self.logger and getattr(self.trainer, "is_global_zero", True):
            B, T = trg["images"].shape[:2]
            gt01 = trg["images"][0].float()
            gt01 = (gt01 / 255.0 if gt01.max() > 1.5 else gt01).clamp(0, 1)
            pred01 = pred.view(B, T, 3, gt01.shape[-2], gt01.shape[-1])[0]
            rgb = torch.cat([pred01, gt01], dim=-1)
            self.logger.experiment.add_video(f"videos/{tag}_rgb_pred_gt", rgb.unsqueeze(0), self.global_step, fps=fps)
        return psnr

    def validation_step(self, batch, batch_idx):
        return self.eval_psnr(batch, log_video_n=(batch_idx < 6), tag="val")

    # ------------------------------------------------------------------
    # Evaluation (mode=test): same metrics/benchmark/output contract as the
    # ZPressor/MVSplat/DepthSplat ModelWrapper.test_step. Reuses the upstream
    # compute_psnr/ssim/lpips, Benchmarker, and image IO from the dataset repo.
    # ------------------------------------------------------------------
    def _load_eval_utils(self):
        from ..dataset.data_module import add_repo_to_path

        if self.upstream_repo_root is not None:
            add_repo_to_path(self.upstream_repo_root, repo_name="upstream-eval")
        from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips
        from src.misc.benchmarker import Benchmarker
        from src.misc.image_io import save_image, save_video
        return compute_psnr, compute_ssim, compute_lpips, Benchmarker, save_image, save_video

    def on_test_start(self) -> None:
        # Evaluate at full capacity. Set here (after the checkpoint load, like
        # on_validation_start) so it does not depend on the caller setting the
        # stage before trainer.test(). stage/mix are not checkpointed.
        self.model.set_stage(self.final_stage, mix=1.0)
        (
            self._compute_psnr,
            self._compute_ssim,
            self._compute_lpips,
            Benchmarker,
            self._save_image,
            self._save_video,
        ) = self._load_eval_utils()
        self.benchmarker = Benchmarker()
        self.test_step_outputs = {"psnr": [], "ssim": [], "lpips": []}
        self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        inputs, trg = batch["inputs"], batch["targets"]
        scene_info = batch.get("scene_info", {}) or {}

        B, T, _, H, W = trg["images"].shape
        assert B == 1, "test_step expects batch size 1 (one scene per step)."

        # Encode -> Gaussians (timed like the upstream encoder).
        with self.benchmarker.time("encoder"):
            preds = self.model(inputs)
        # Decode/render to the target views (timed like the upstream decoder).
        with self.benchmarker.time("decoder", num_calls=T):
            out = render_static_batched(preds, trg, render_depth=False)

        pred = out["img"].view(B, T, 3, H, W)[0].clamp(0, 1)   # [T,3,H,W]
        gt = trg["images"][0].float().clamp(0, 1)              # [T,3,H,W]

        scene = scene_info.get("scene", [f"scene_{batch_idx:06d}"])
        scene = scene[0] if isinstance(scene, (list, tuple)) else scene
        out_path = self._test_out_dir()

        if self.test_cfg.get("save_input_images", False):
            for idx, color in zip(inputs["frame_ids"][0], inputs["images"][0]):
                self._save_image(color, out_path / scene / f"input/{int(idx):0>6}.png")
        if self.test_cfg.get("save_image", False):
            for idx, color in zip(trg["frame_ids"][0], pred):
                self._save_image(color, out_path / scene / f"color/{int(idx):0>6}.png")
        if self.test_cfg.get("save_gt_image", False):
            for idx, g in zip(trg["frame_ids"][0], gt):
                self._save_image(g, out_path / scene / f"gt/{int(idx):0>6}_gt.png")
        if self.test_cfg.get("save_video", False):
            self._save_video([f for f in pred], out_path / "video" / f"{scene}.mp4")

        if self.test_cfg.get("compute_scores", True):
            if batch_idx < int(self.test_cfg.get("eval_time_skip_steps", 0)):
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += T
            self.test_step_outputs["psnr"].append(self._compute_psnr(gt, pred).mean().item())
            self.test_step_outputs["ssim"].append(self._compute_ssim(gt, pred).mean().item())
            self.test_step_outputs["lpips"].append(self._compute_lpips(gt, pred).mean().item())

    def _test_out_dir(self):
        from pathlib import Path

        return Path(self.test_cfg.get("output_path", "./outputs/test")) / self.experiment_name

    def on_test_end(self) -> None:
        import json

        out_dir = self._test_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = {}

        if self.test_cfg.get("compute_scores", True):
            for name, scores in self.test_step_outputs.items():
                if not scores:
                    continue
                avg = sum(scores) / len(scores)
                saved[name] = avg
                print(f"{name}: {avg:.4f}")
                with (out_dir / f"scores_{name}_all.json").open("w") as f:
                    json.dump(scores, f)

        # Timing (skip warm-up steps), matching the upstream benchmark dump.
        try:
            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict.get(tag, 0)):]
                if times:
                    saved[tag] = [len(times), float(sum(times) / len(times))]
                    print(f"{tag}: {len(times)} calls, avg. {saved[tag][1]:.4f} s/call")
            self.benchmarker.dump(out_dir / "benchmark.json")
            if torch.cuda.is_available():
                self.benchmarker.dump_memory(out_dir / "peak_memory.json")
        except Exception as exc:  # benchmarker is best-effort
            print(f"[test] benchmark dump skipped: {exc}")

        with (out_dir / "scores_all_avg.json").open("w") as f:
            json.dump(saved, f)

    def configure_optimizers(self):
        # Optimizer + LR schedule (and the weight-decay param-group policy) live
        # in model.optim so they can be reused / tested without a LightningModule.
        return build_optimizer_and_scheduler(
            self,
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
            total_steps=getattr(getattr(self, "trainer", None), "estimated_stepping_batches", None),
            warmup_pct=self.warmup_pct,
            min_lr_ratio=self.min_lr_ratio,
            min_lr_floor=self.min_lr_floor,
            warmup_epochs_fallback=self.warmup_epochs_fallback,
            max_epochs=int(getattr(getattr(self, "trainer", None), "max_epochs", 100)),
        )



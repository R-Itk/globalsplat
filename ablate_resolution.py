#!/usr/bin/env python
"""Resolution ablation for GlobalSplat -- one file, every input-side variant.

All variants share the same idea: load the dataset at the RENDER resolution (so the
target GT and the rasterized output are full-res), then transform ONLY the context
views at the encoder seam (GlobalSplat.forward), leaving targets/rendering untouched.
The render is always at --resolution; the modes differ in what the encoder sees.

MODES (--mode)
--------------
  plain        Context = render resolution. No transform.
               --resolution 512  -> naive 512-in / 512-out (the OOD penalty case).
               --resolution 256  -> the native trained-res baseline.

  render-only  Context downsampled to --context-resolution (intrinsics rescaled,
               frustum exact); render stays at --resolution.
               The legitimate high-res-output mode: input 256, render 512.

  virtual      Context split into --phase**2 POLYPHASE virtual cameras at
               resolution/phase via interlaced subsampling, principal point shifted
               so rays are the exact originals. In-distribution tokens, all pixels
               kept. (--resolution 512 --phase 2 -> 4 x 256 virtual cams.)

  duplicate    Each context view repeated --duplicate times (identical content, more
               tokens). The control: softmax is invariant to duplicate keys, so this
               should be ~a no-op -- use it to confirm token COUNT is not the issue.

Render is always at --resolution. Pick resolution/{phase or context-res} == 256
(the training res) for the in-distribution sweet spot.

EXAMPLES
--------
    R=512; IDX=assets/eval_index/re10k/c3g_re10k_ctx_12v_trg_3v.json
    CK=globalsplat-re10k-32k.ckpt   # Tables 2/3 in the paper use the 32K model

    python ablate_resolution.py --ckpt $CK --mode plain       --resolution $R --overrides dataset.eval_index_path=$IDX
    python ablate_resolution.py --ckpt $CK --mode render-only  --resolution $R --context-resolution 256 --overrides dataset.eval_index_path=$IDX
    python ablate_resolution.py --ckpt $CK --mode virtual      --resolution $R --phase 2 --overrides dataset.eval_index_path=$IDX
    python ablate_resolution.py --ckpt $CK --mode duplicate    --resolution $R --duplicate 4 --overrides dataset.eval_index_path=$IDX

    # sweep all of them, logging each:
    for M in plain render-only virtual duplicate; do
      echo "=== $M ==="
      python ablate_resolution.py --ckpt $CK --mode $M --resolution $R \
          --overrides dataset.eval_index_path=$IDX 2>&1 | tee ablate_${M}.log
    done

Reading it: `plain @512` is the penalty floor; `render-only` recovers it (input stays
in-distribution); `virtual` tests whether polyphase beats that recovery; `duplicate`
should match `plain @512` minus noise (token count is irrelevant). All 512 numbers
share a small fixed loss from the GT being upsampled from RE10K's native ~360p.

Requires the loader to deliver --resolution context (upstream crop asserts
target<=source); ~360p data will assert at 512 regardless of mode.

See docs/EVALUATION.md ("High-resolution input & resolution-scaling ablation")
for how these modes map to the paper's Tables 2-3.
"""
import argparse
import os

import torch
import torch.nn.functional as F


_ANNOUNCED = False


def _announce(msg: str):
    global _ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(">> " + msg, flush=True)


# ---------------------------------------------------------------------------
# Context transforms. Each takes/returns the model batch (post scene-norm +
# pixel intrinsics):
#   images [B,V,3,H,W] raw [0,1] | intrinsic [B,V,3,3] pixel | extrinsic/c2w [B,V,4,4]
#   frame_ids [B,V]. Only context is touched; targets/rendering are separate.
# ---------------------------------------------------------------------------

def _tile_views(inputs, out, V, K, skip=("images", "intrinsic")):
    """Tile every per-view field (shape [B,V,...]) K times along the view dim."""
    for k, val in inputs.items():
        if k in skip or k in out:
            continue
        if torch.is_tensor(val) and val.ndim >= 2 and val.shape[1] == V:
            out[k] = val.repeat(1, K, *([1] * (val.ndim - 2)))
    return out


def downsample_context(inputs, ctx_res: int, resample: str = "bilinear"):
    """render-only: shrink context to ctx_res, rescale intrinsics (frustum exact)."""
    img, K = inputs["images"], inputs["intrinsic"]
    B, V, C, H, W = img.shape
    if H == ctx_res and W == ctx_res:
        return inputs
    sx, sy = W / float(ctx_res), H / float(ctx_res)
    x = img.reshape(B * V, C, H, W)
    if resample == "area":
        x = F.interpolate(x, size=(ctx_res, ctx_res), mode="area")
    else:
        x = F.interpolate(x, size=(ctx_res, ctx_res), mode=resample,
                          align_corners=False, antialias=True)
    out = dict(inputs)
    out["images"] = x.reshape(B, V, C, ctx_res, ctx_res)
    Kp = K.clone()
    Kp[..., 0, 0] /= sx; Kp[..., 0, 2] /= sx
    Kp[..., 1, 1] /= sy; Kp[..., 1, 2] /= sy
    out["intrinsic"] = Kp
    _announce(f"context downsampled {H}x{W} -> {ctx_res}x{ctx_res} ({resample}); "
              f"targets/render unchanged")
    return out


def _box_prefilter(img, s):
    B, V, C, H, W = img.shape
    x = img.reshape(B * V, C, H, W)
    x = F.pad(x, (0, s - 1, 0, s - 1), mode="replicate")
    x = F.avg_pool2d(x, kernel_size=s, stride=1)
    return x.reshape(B, V, C, H, W)


def make_virtual_cameras(inputs, phase: int = 2, prefilter: bool = False):
    """virtual: each view -> phase**2 polyphase virtual cams, exact original rays."""
    s = int(phase)
    if s <= 1:
        return inputs
    img, K = inputs["images"], inputs["intrinsic"]
    B, V, C, H, W = img.shape
    assert H % s == 0 and W % s == 0, f"H,W ({H},{W}) must be divisible by phase {s}"
    Hs, Ws = H // s, W // s
    assert Hs % 8 == 0 and Ws % 8 == 0, (
        f"sub-view {Hs}x{Ws} must stay divisible by patch_size=8 "
        f"(pick resolution/phase a multiple of 8, ideally 256)")
    src = _box_prefilter(img, s) if prefilter else img
    fx, fy = K[..., 0, 0], K[..., 1, 1]
    cx, cy = K[..., 0, 2], K[..., 1, 2]
    imgs, Ks = [], []
    for py in range(s):
        for px in range(s):
            imgs.append(src[..., py::s, px::s])
            Kp = K.clone()
            Kp[..., 0, 0] = fx / s
            Kp[..., 1, 1] = fy / s
            Kp[..., 0, 2] = 0.5 + (cx - px - 0.5) / s
            Kp[..., 1, 2] = 0.5 + (cy - py - 0.5) / s
            Ks.append(Kp)
    K2 = s * s
    out = dict(inputs)
    out["images"] = torch.cat(imgs, dim=1)
    out["intrinsic"] = torch.cat(Ks, dim=1)
    for key in ("extrinsic", "c2w"):
        if key in inputs and torch.is_tensor(inputs[key]):
            out[key] = inputs[key].repeat(1, K2, *([1] * (inputs[key].ndim - 2)))
    if "frame_ids" in inputs and torch.is_tensor(inputs["frame_ids"]):
        out["frame_ids"] = inputs["frame_ids"].repeat(1, K2)
    _tile_views(inputs, out, V, K2)
    _announce(f"virtual cameras: {V} ctx view(s) x {K2} phases = {V*K2} views, "
              f"each {Hs}x{Ws} (from {H}x{W}); prefilter={prefilter}")
    return out


def duplicate_context(inputs, k: int = 4):
    """duplicate (control): repeat each context view k times, identical content."""
    k = int(k)
    if k <= 1:
        return inputs
    img = inputs["images"]
    B, V = img.shape[:2]
    out = dict(inputs)
    out["images"] = img.repeat(1, k, *([1] * (img.ndim - 2)))
    out["intrinsic"] = inputs["intrinsic"].repeat(1, k, 1, 1)
    for key in ("extrinsic", "c2w"):
        if key in inputs and torch.is_tensor(inputs[key]):
            out[key] = inputs[key].repeat(1, k, *([1] * (inputs[key].ndim - 2)))
    if "frame_ids" in inputs and torch.is_tensor(inputs["frame_ids"]):
        out["frame_ids"] = inputs["frame_ids"].repeat(1, k)
    _tile_views(inputs, out, V, k)
    _announce(f"duplicate control: {V} ctx view(s) repeated x{k} = {V*k} views "
              f"(identical content)")
    return out


# ---------------------------------------------------------------------------

def build_transform(args):
    m = args.mode
    if m == "plain":
        return None
    if m == "render-only":
        ctx, rs = int(args.context_resolution), args.resample
        return lambda inp: downsample_context(inp, ctx, rs)
    if m == "virtual":
        ph, pf = int(args.phase), bool(args.prefilter)
        return lambda inp: make_virtual_cameras(inp, ph, pf)
    if m == "duplicate":
        d = int(args.duplicate)
        return lambda inp: duplicate_context(inp, d)
    raise SystemExit(f"[ablate] unknown mode {m!r}")


def validate(args):
    res = int(args.resolution)
    if res % 8 != 0:
        raise SystemExit(f"[ablate] resolution must be divisible by 8; got {res}")
    if args.mode == "render-only":
        ctx = int(args.context_resolution)
        if ctx % 8 != 0:
            raise SystemExit(f"[ablate] context-resolution must be divisible by 8; got {ctx}")
        if ctx > res:
            raise SystemExit(f"[ablate] context-resolution ({ctx}) must be <= resolution ({res})")
    if args.mode == "virtual":
        s = int(args.phase)
        if s < 1:
            raise SystemExit("[ablate] phase must be >= 1")
        if s > 1 and (res % s != 0 or (res // s) % 8 != 0):
            raise SystemExit(f"[ablate] need resolution%phase==0 and resolution/phase%8==0 "
                             f"(got resolution={res}, phase={s})")
    if args.mode == "duplicate" and int(args.duplicate) < 1:
        raise SystemExit("[ablate] duplicate must be >= 1")


def parse_args():
    p = argparse.ArgumentParser(
        description="Resolution ablation for GlobalSplat (plain / render-only / virtual / duplicate).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="Path to the .ckpt to evaluate.")
    p.add_argument("--mode", required=True,
                   choices=["plain", "render-only", "virtual", "duplicate"],
                   help="Which input-side variant to run.")
    p.add_argument("--resolution", type=int, default=512,
                   help="Render/target resolution (GT res, dataset load res). Div by 8.")
    p.add_argument("--context-resolution", type=int, default=256, dest="context_resolution",
                   help="[render-only] resolution context is downsampled to. Div by 8.")
    p.add_argument("--phase", type=int, default=2,
                   help="[virtual] subsample factor s; each view -> s*s virtual cams.")
    p.add_argument("--prefilter", action="store_true",
                   help="[virtual] box low-pass before decimation (breaks exact rays).")
    p.add_argument("--duplicate", type=int, default=4,
                   help="[duplicate] times to repeat each context view.")
    p.add_argument("--resample", default="bilinear", choices=["bilinear", "bicubic", "area"],
                   help="[render-only] interpolation for the context downsample.")
    p.add_argument("--experiment", default="re10k_32k",
                   help="Experiment config name (config/experiment/<name>.yaml).")
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs to use.")
    p.add_argument("--overrides", nargs="*", default=[],
                   help="Extra Hydra overrides, e.g. dataset.eval_index_path=... key=value ...")
    return p.parse_args()


def main():
    args = parse_args()
    validate(args)

    from pathlib import Path
    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_file():
        raise SystemExit(f"[ablate] checkpoint not found: {ckpt}")

    transform = build_transform(args)

    # Patch the encoder seam so the model the repo builds applies the chosen context
    # transform right before tokenization (test_step -> self.model(inputs) ->
    # GlobalSplat.forward). Targets and rendering are untouched.
    import globalsplat.model.globalsplat as gs_mod
    _orig_forward = gs_mod.GlobalSplat.forward

    if transform is None:
        def _patched_forward(self, inputs):
            _announce(f"plain: context = render resolution ({args.resolution}); no transform")
            return _orig_forward(self, inputs)
    else:
        def _patched_forward(self, inputs):
            return _orig_forward(self, transform(inputs))
    gs_mod.GlobalSplat.forward = _patched_forward

    res = int(args.resolution)
    overrides = [
        f"+experiment={args.experiment}",
        "mode=test",
        f"checkpointing.load={ckpt.resolve()}",
        f"dataset.image_shape=[{res},{res}]",
        f"trainer.devices={int(args.devices)}",
        *args.overrides,
    ]

    detail = {
        "plain": f"context={res}",
        "render-only": f"context={args.context_resolution} (resample={args.resample})",
        "virtual": f"phase={args.phase} -> {args.phase**2} x {res // max(args.phase,1)}"
                   f"{' +prefilter' if args.prefilter else ''}",
        "duplicate": f"x{args.duplicate} (control)",
    }[args.mode]
    print(f">> Evaluating {ckpt}", flush=True)
    print(f">> mode={args.mode}  render={res}  {detail}  devices={args.devices}", flush=True)

    import globalsplat
    from hydra import initialize_config_dir, compose
    from globalsplat.main import main as run_globalsplat

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(globalsplat.__file__)))
    config_dir = os.path.join(repo_root, "config")
    if not os.path.isdir(config_dir):
        raise SystemExit(f"[ablate] could not find config dir at {config_dir}")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="main", overrides=overrides)

    run_globalsplat(cfg)


if __name__ == "__main__":
    main()
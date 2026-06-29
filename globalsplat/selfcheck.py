"""Post-install self-check for GlobalSplat.

Run this right after ``bash install.sh`` (or ``pip install -e .``) to verify the
environment is wired up correctly, end to end:

    python -m globalsplat.selfcheck

It runs, in order:
  1. environment       - report torch / CUDA / key dependency versions
  2. forward (CPU)     - full model forward on a tiny synthetic batch; checks the
                         Gaussian output shapes + finiteness at stage 0 and 3
  3. backward (CPU)    - one loss.backward(); checks finite gradients
  4. config compose    - best-effort: compose the shipped Hydra config
                         (``main`` + ``+experiment=re10k_16k``); SKIP if the
                         ``config/`` tree is not reachable from the CWD
  5. render (GPU)      - if gsplat + a CUDA GPU are present, rasterize one frame;
                         SKIP otherwise

No dataset, no checkpoint, and no GPU are required for the core checks (1-4).
Each check prints PASS / SKIP / FAIL; the process exits non-zero if any FAILs.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PASS, SKIP, FAIL = "PASS", "SKIP", "FAIL"
_MARK = {PASS: "✓", SKIP: "–", FAIL: "✗"}


def _fmt(status: str, name: str, detail: str = "") -> str:
    line = f"  [{_MARK[status]}] {status:4} {name}"
    return f"{line}  ({detail})" if detail else line


# --- tiny synthetic model/inputs --------------------------------------------

def _tiny_model():
    # Small dims so the check is fast; dim_latents divisible by num_heads.
    from globalsplat.model.globalsplat import GlobalSplat

    return GlobalSplat(
        sh_degree=3,
        static_only=True,
        patch_size=8,
        latent_rep_token_amount=32,
        dim_latents=64,
        dim_rays=16,
        dim_rgb_feat=16,
        rounds=2,
        slot_calib_layers_per_round=1,
        num_heads=4,
    )


def _tiny_inputs(B=1, V=2, H=16, W=16, device="cpu"):
    import torch

    images = torch.rand(B, V, 3, H, W, device=device)
    # Pixel-space intrinsics with the principal point inside the image and
    # fx, fy > 0 (compute_rays asserts this).
    K = torch.zeros(B, V, 3, 3, device=device)
    K[..., 0, 0] = 10.0
    K[..., 1, 1] = 10.0
    K[..., 0, 2] = W / 2.0
    K[..., 1, 2] = H / 2.0
    K[..., 2, 2] = 1.0
    c2w = torch.eye(4, device=device).reshape(1, 1, 4, 4).repeat(B, V, 1, 1)
    return {"images": images, "intrinsic": K, "c2w": c2w}


# --- individual checks (each returns (status, detail) or raises) -------------

def check_environment():
    import torch

    extras = []
    for mod in ("pytorch_lightning", "hydra", "einops", "gsplat"):
        try:
            m = __import__(mod)
            extras.append(f"{mod} {getattr(m, '__version__', '?')}")
        except Exception:
            extras.append(f"{mod} —")
    detail = (
        f"torch {torch.__version__}, CUDA={'yes' if torch.cuda.is_available() else 'no'}; "
        + ", ".join(extras)
    )
    return PASS, detail


def check_forward():
    import torch

    from globalsplat.model.types import Gaussians

    torch.manual_seed(0)
    model = _tiny_model().eval()
    inputs = _tiny_inputs()
    for stage in (0, 3):
        model.set_stage(stage, mix=1.0)
        with torch.no_grad():
            out = model(inputs)
        assert isinstance(out, Gaussians), f"expected Gaussians, got {type(out)}"
        n = 32 * (2 ** stage)  # decoder emits 2**stage splats per token
        assert out.means.shape == (1, n, 3), f"means {tuple(out.means.shape)} != (1,{n},3)"
        for name, t in (
            ("means", out.means),
            ("scales", out.scales),
            ("rotations", out.rotations),
            ("opacities", out.opacities),
            ("sh", out.sh),
            ("reg", out.reg),
        ):
            assert torch.isfinite(t).all(), f"{name} contains non-finite values"
        assert (out.scales > 0).all(), "scales must be positive (post-exp)"
    return PASS, "stages 0 & 3: shapes + finiteness ok"


def check_backward():
    import torch

    torch.manual_seed(0)
    model = _tiny_model().train()
    model.set_stage(2, mix=0.5)
    out = model(_tiny_inputs())
    (out.means.square().mean() + out.reg).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(
        g is not None and torch.isfinite(g).all() for g in grads
    ), "no finite gradients after backward()"
    return PASS, "loss.backward() produced finite gradients"


def check_config():
    cfg_dir = Path.cwd() / "config"
    if not (cfg_dir / "main.yaml").exists():
        return SKIP, "config/ not found in CWD (run from the repo root to test)"
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name="main", overrides=["+experiment=re10k_16k"])
    assert cfg.model is not None, "composed cfg has no `model`"
    assert cfg.dataset is not None, "composed cfg has no `dataset`"
    return PASS, "main + experiment=re10k_16k composed ok"


def check_render():
    import torch

    try:
        import gsplat  # noqa: F401
    except Exception:
        return SKIP, "gsplat not installed (GPU-only [gpu] extra)"
    if not torch.cuda.is_available():
        return SKIP, "no CUDA GPU available"

    from globalsplat.model.rendering import render_static_batched

    torch.manual_seed(0)
    model = _tiny_model().eval().cuda()
    model.set_stage(3, mix=1.0)
    with torch.no_grad():
        gaussians = model(_tiny_inputs(device="cuda"))
    H = W = 16
    inp = _tiny_inputs(H=H, W=W, device="cuda")
    meta = {
        "images": torch.zeros(1, 1, 3, H, W, device="cuda"),
        "intrinsic": inp["intrinsic"][:, :1],            # [1,1,3,3]
        "extrinsic": torch.eye(4, device="cuda").reshape(1, 1, 4, 4),  # w2c
    }
    out = render_static_batched(gaussians, meta, render_depth=True)
    assert torch.isfinite(out["img"]).all(), "render produced non-finite image"
    return PASS, f"rendered {tuple(out['img'].shape)} on GPU"


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m globalsplat.selfcheck",
        description="Post-install sanity check for the GlobalSplat environment.",
    )
    ap.add_argument(
        "--no-render", action="store_true",
        help="skip the optional GPU render check (checks 1-4 only)",
    )
    args = ap.parse_args()

    print("GlobalSplat self-check")
    print("=" * 60)

    checks = [
        ("environment", check_environment),
        ("model forward (CPU)", check_forward),
        ("model backward (CPU)", check_backward),
        ("hydra config compose", check_config),
    ]
    if not args.no_render:
        checks.append(("gsplat render (GPU)", check_render))

    statuses = []
    for name, fn in checks:
        try:
            status, detail = fn()
        except Exception as exc:
            status, detail = FAIL, f"{type(exc).__name__}: {exc}"
            print(_fmt(status, name, detail))
            traceback.print_exc()
            statuses.append(status)
            continue
        print(_fmt(status, name, detail))
        statuses.append(status)

    print("=" * 60)
    n_pass, n_skip, n_fail = (statuses.count(s) for s in (PASS, SKIP, FAIL))
    print(f"{n_pass} passed, {n_skip} skipped, {n_fail} failed")
    if n_fail:
        print("SELF-CHECK FAILED")
        return 1
    print("SELF-CHECK OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""(Maintainer tool) Build the bundled demo scenes from the RE10K dataset.

For each requested scene this dumps a self-contained, runnable scene folder under
``demo/scenes/<name>/``:

    images/000.png ...      the input context views (256x256), spaced >= --min-gap frames apart
    cameras.json            per-view pixel intrinsics + normalized c2w
    <name>.ply              precomputed GlobalSplat output (for instant viewing)

Context views are picked evenly across each clip with a minimum frame gap (wide
baseline -> visible parallax) via a generated ``demo/demo_index.json``. End users
then run ``demo/run_demo.py`` on these folders without the dataset. Requires the
RE10K chunks + the model install.

    python demo/export_demo.py --min-gap 10 --max-views 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

import numpy as np
import torch
from torchvision.utils import save_image
from hydra import compose, initialize_config_dir

from globalsplat.main import build_datamodule, build_model
from globalsplat.model.rendering import _rot_to_quat
from ply_io import write_ply

# RE10K test scenes used for the demo (chosen for length / content).
DEFAULT_SCENES = ["5aca87f95a9412c6", "322261824c4a3003", "ffa95c3b40609c76"]


def build_spaced_index(scene_keys, data_root: Path, min_gap: int, max_views: int) -> dict:
    """For each scene, pick <=max_views context frames spaced >= min_gap apart
    (evenly across the clip), plus a few spread-out targets. Returns the eval-index
    dict {scene: {context, target}} the upstream 'evaluation' sampler reads."""
    test_root = data_root / "test"
    key_to_chunk = json.loads((test_root / "index.json").read_text())
    index = {}
    for k in scene_keys:
        chunk = torch.load(test_root / key_to_chunk[k], weights_only=False)
        ex = next(e for e in chunk if e["key"] == k)
        N = len(ex["images"])
        n = min(max_views, (N - 1) // min_gap + 1)
        ctx = sorted(set(int(round(x)) for x in np.linspace(0, N - 1, n)))
        tgt = sorted(set(int(round(x)) for x in np.linspace(N // 6, N - 1 - N // 6, 3)))
        gaps = [b - a for a, b in zip(ctx, ctx[1:])]
        print(f"  {k}: N={N} -> {len(ctx)} context views, min gap {min(gaps)}", flush=True)
        index[k] = {"context": ctx, "target": tgt}
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default="globalsplat-re10k-16k-noopacity.ckpt")
    ap.add_argument("--experiment", default="re10k_16k")
    ap.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    ap.add_argument("--min-gap", type=int, default=10, help="minimum frames between context views")
    ap.add_argument("--max-views", type=int, default=12)
    ap.add_argument("--out", default="demo/scenes")
    args = ap.parse_args()

    with initialize_config_dir(config_dir=str(_ROOT / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=[
            f"+experiment={args.experiment}", "dataset=re10k_eval_ctx12",
            "mode=test", "optimizer.batch_size=1",
        ])

    data_root = Path(cfg.dataset.dataset_roots[0])
    if not data_root.is_absolute():
        data_root = (_ROOT / data_root).resolve()
    print("building spaced demo index:", flush=True)
    index = build_spaced_index(args.scenes, data_root, args.min_gap, args.max_views)
    index_path = _ROOT / "demo" / "demo_index.json"
    index_path.write_text(json.dumps(index, indent=2))
    # Point the eval sampler at our index; non-listed scenes are skipped (ValueError).
    cfg.dataset.eval_index_path = str(index_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.model).to(device).eval()
    model.set_stage(cfg.curriculum.final_stage, mix=1.0)
    sd = torch.load(_ROOT / args.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    print(f"loaded {args.checkpoint} (stage {cfg.curriculum.final_stage}, M_max={cfg.model.get('M_max')})", flush=True)

    dm, _ = build_datamodule(cfg)
    dm.setup("test")
    loader = dm.test_dataloader()

    out_root = _ROOT / args.out
    wanted = set(args.scenes)
    done = set()
    for batch in loader:
        s = batch["scene_info"]["scene"]
        name = (s[0] if isinstance(s, (list, tuple)) else s)
        if name not in wanted or name in done:
            continue
        done.add(name)
        scene_dir = out_root / name
        # fresh image dir (view count may have changed)
        img_dir = scene_dir / "images"
        if img_dir.exists():
            for p in img_dir.glob("*.png"):
                p.unlink()
        img_dir.mkdir(parents=True, exist_ok=True)

        imgs = batch["inputs"]["images"][0]
        Ks = batch["inputs"]["intrinsic"][0]
        c2ws = batch["inputs"]["c2w"][0]
        V, _, H, W = imgs.shape
        views = []
        for v in range(V):
            fn = f"images/{v:03d}.png"
            save_image(imgs[v], scene_dir / fn)
            views.append({"image": fn, "intrinsic": Ks[v].tolist(), "c2w": c2ws[v].tolist()})
        (scene_dir / "cameras.json").write_text(json.dumps({
            "dataset": "re10k", "scene": name, "image_shape": [H, W],
            "checkpoint": Path(args.checkpoint).name, "experiment": args.experiment,
            "context_frames": index[name]["context"], "views": views,
        }, indent=2))

        inputs = {k: v.to(device) for k, v in batch["inputs"].items() if torch.is_tensor(v)}
        with torch.no_grad():
            g = model(inputs)[0]
        quats = _rot_to_quat(g.rotations.unsqueeze(0))[0]
        n = write_ply(scene_dir / f"{name}.ply", g.means.cpu(), g.scales.cpu(),
                      quats.cpu(), g.sh.cpu(), g.opacities.cpu())
        print(f"  {scene_dir.relative_to(_ROOT)}  ({V} views, {n} gaussians)", flush=True)
        if done >= wanted:
            break

    print(f"\nwrote {len(done)}/{len(wanted)} scene folder(s) under {args.out}")


if __name__ == "__main__":
    main()

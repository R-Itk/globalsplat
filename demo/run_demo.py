#!/usr/bin/env python3
"""Run GlobalSplat on a bundled demo scene and export a 3DGS ``.ply``.

Self-contained: needs only this repo, a checkpoint, and a scene folder under
``demo/scenes/<name>/`` (``images/`` + ``cameras.json``). No RE10K dataset
required — the camera parameters are bundled with the scene.

    python demo/run_demo.py --scene demo/scenes/<name> \
        --checkpoint globalsplat-re10k-16k-noopacity.ckpt

Download the checkpoint from the Pretrained models table in the top-level README
(Hugging Face); the demo scenes were produced with the 16K no-opacity model.

Writes ``<name>.ply`` inside the scene folder; open it in a web splat viewer
(see demo/README.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))      # so `globalsplat` imports work from any CWD
sys.path.insert(0, str(_HERE))      # so `ply_io` (sibling) imports work

import torch
from torchvision.io import read_image
from hydra import compose, initialize_config_dir

from globalsplat.main import build_model
from globalsplat.model.rendering import _rot_to_quat
from ply_io import write_ply


def load_scene(scene_dir: Path, device: str):
    cam_file = scene_dir / "cameras.json"
    if not cam_file.exists():
        raise SystemExit(
            f"no cameras.json in {scene_dir} — pass --scene a demo/scenes/<name>/ folder "
            f"(each has images/ + cameras.json)."
        )
    cams = json.loads(cam_file.read_text())
    imgs, Ks, c2ws = [], [], []
    for v in cams["views"]:
        im = read_image(str(scene_dir / v["image"]))[:3].float() / 255.0   # [3,H,W] in [0,1]
        imgs.append(im)
        Ks.append(torch.tensor(v["intrinsic"], dtype=torch.float32))
        c2ws.append(torch.tensor(v["c2w"], dtype=torch.float32))
    images = torch.stack(imgs)[None].to(device)        # [1, V, 3, H, W]
    intrinsic = torch.stack(Ks)[None].to(device)       # [1, V, 3, 3]
    c2w = torch.stack(c2ws)[None].to(device)           # [1, V, 4, 4]
    return {"images": images, "intrinsic": intrinsic, "c2w": c2w}, cams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True, help="path to demo/scenes/<name>/")
    ap.add_argument("--checkpoint", default="globalsplat-re10k-16k-noopacity.ckpt",
                    help="path to a downloaded checkpoint (see the README Pretrained models table)")
    ap.add_argument("--experiment", default="re10k_16k",
                    help="model config matching the checkpoint (re10k_2k / re10k_16k / re10k_32k)")
    ap.add_argument("--out", default=None, help="output .ply (default: <scene>/<name>.ply)")
    args = ap.parse_args()

    scene_dir = Path(args.scene).resolve()
    out = Path(args.out) if args.out else scene_dir / f"{scene_dir.name}.ply"

    with initialize_config_dir(config_dir=str(_ROOT / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=[f"+experiment={args.experiment}", "mode=test"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.model).to(device).eval()
    model.set_stage(cfg.curriculum.final_stage, mix=1.0)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt} — download it from the README Pretrained models table.")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint/config mismatch: missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}")

    inputs, cams = load_scene(scene_dir, device)
    print(f"scene '{scene_dir.name}': {inputs['images'].shape[1]} views, "
          f"checkpoint {ckpt.name}, stage {cfg.curriculum.final_stage} (M_max={cfg.model.get('M_max')})")

    with torch.no_grad():
        g = model(inputs)[0]
    quats = _rot_to_quat(g.rotations.unsqueeze(0))[0]
    n = write_ply(out, g.means.cpu(), g.scales.cpu(), quats.cpu(), g.sh.cpu(), g.opacities.cpu())
    print(f"wrote {out}  ({n} gaussians, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

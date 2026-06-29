# Installation

**Requirements:** Linux, Python 3.10, a CUDA 12.x toolkit with `nvcc` on `PATH`,
and an NVIDIA GPU (Ampere+ recommended). The one-command path below covers the
common case; the rest of this file is for edge cases (no compiler, mixed-GPU
clusters, torch 2.4).

## One command

```bash
git clone <this-repo> globalsplat && cd globalsplat
python -m venv .venv && source .venv/bin/activate

# Run on a node with a CUDA 12.x toolkit (e.g. `module load cuda/12.1`).
# Installs torch 2.5.1 (cu121) + core deps, then compiles gsplat from source.
# The target GPU arch (gsplat "sm_" / TORCH_CUDA_ARCH_LIST) is auto-detected
# from the node's GPUs -- no need to look it up.
bash install.sh
```

Verify:

```bash
python -c "import torch, gsplat; print(torch.version.cuda, gsplat.__version__)"
```

## Choosing the GPU architecture

Running on multiple / different GPUs than the build node (e.g. a mixed cluster)?
Choose the arch(s) explicitly so the build covers every device it will run on:

```bash
bash install.sh --arch "8.0;8.6;9.0"   # A100 + A6000/3090 + H100
bash install.sh --arch all             # broad portable set: 7.0;7.5;8.0;8.6;8.9;9.0
bash install.sh --arch "9.0+PTX"       # H100 + PTX so it also JITs on newer archs
# (equivalently: TORCH_CUDA_ARCH_LIST="8.0;9.0" bash install.sh)

bash install.sh --print-arch           # just show what it would use (no install)
```

Precedence: `--arch` > `TORCH_CUDA_ARCH_LIST` env > auto-detect > broad default.
More archs = more portable but a proportionally longer compile.

Arch cheat-sheet: `7.5`=T4/2080  `8.0`=A100  `8.6`=A6000/3090  `8.9`=4090/L40
`9.0`=H100.

## Installing by hand

`install.sh` just runs the three ordered steps below (torch must exist before
gsplat compiles, which is why it can't be a single `pip install`). Do it by hand
to swap versions:

```bash
# 1) torch + torchvision for CUDA 12.1 (from the PyTorch index, not PyPI default)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 2) build tools for the no-isolation gsplat compile
pip install ninja "setuptools<81" wheel numpy==1.26.3

# 3) gsplat compiled FROM SOURCE against the torch above (it lives in the [gpu]
#    extra, not core), so its CUDA kernels (gsplat.csrc) are built now instead of
#    JIT-compiled at first run.
#    .[gpu]                : pull the pinned gsplat (kept out of core deps)
#    --no-binary gsplat    : force a source build (PyPI ships a JIT build otherwise)
#    --no-build-isolation  : build against your torch, not a freshly pulled cu13x one
#    needs nvcc on PATH whose CUDA major matches torch (both 12.x), e.g. module load cuda/12.1
TORCH_CUDA_ARCH_LIST="8.0" pip install -e ".[gpu]" --no-build-isolation --no-binary gsplat
```

`TORCH_CUDA_ARCH_LIST` is auto-detected by `install.sh`; set it explicitly only to
cross-compile for a GPU that isn't in the build box (e.g. building on a login node).

## No CUDA compiler on the box?

gsplat's prebuilt-wheel index tops out at **torch 2.4** (`pt24cu121` / `pt24cu124`,
cp310 + linux only -- there is no `pt25`). If you can use torch 2.4 instead of 2.5,
pip can fetch a wheel with no compilation:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[gpu]" --extra-index-url https://docs.gsplat.studio/whl/pt24cu121
```

(The wheel's CUDA must match torch's -- both cu121 here.)

## Perceptual-loss weights (training only)

The VGG-19 perceptual loss expects `metric_checkpoint/imagenet-vgg-verydeep-19.mat`.
Download it once (the loss will otherwise try to fetch it at runtime):

```bash
mkdir -p metric_checkpoint
wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat \
  -O metric_checkpoint/imagenet-vgg-verydeep-19.mat
```

## Still stuck?

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for the common gsplat / CUDA build
and runtime errors and their fixes.

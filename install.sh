#!/usr/bin/env bash
#
# One-command install for GlobalSplat.
#   - keeps torch 2.5.1 (built for CUDA 12.1, matching the released model)
#   - compiles gsplat's CUDA kernels FROM SOURCE at install time, so there is no
#     first-run JIT ("Setting up CUDA with MAX_JOBS...") and no DDP/NFS build race.
#
# Choosing the target GPU arch (the gsplat "sm_" / TORCH_CUDA_ARCH_LIST):
#   * Default: detected automatically from the GPUs on THIS node.
#   * Multiple/other devices: choose explicitly so the build runs everywhere it
#     will be deployed (e.g. a heterogeneous cluster, or building on a login node):
#         bash install.sh --arch "8.0;8.6;9.0"     # A100 + A6000/3090 + H100
#         bash install.sh --arch all               # broad portable set (see below)
#         bash install.sh --arch "9.0+PTX"         # H100 + PTX for forward-compat
#     (or, equivalently, the env var:  TORCH_CUDA_ARCH_LIST="8.0;9.0" bash install.sh)
#   Precedence: --arch  >  TORCH_CUDA_ARCH_LIST env  >  auto-detect  >  broad default.
#   Compiling for more archs is more portable but takes proportionally longer.
#
#   arch cheat-sheet: 7.0=V100  7.5=T4/2080  8.0=A100  8.6=A6000/3090  8.9=4090/L40  9.0=H100
#
# Just want to see which arch would be used (no install)?
#     bash install.sh --print-arch
#     bash install.sh --arch all --print-arch
#
# Run inside your activated venv/conda env, on a node with a CUDA 12.x toolkit
# loaded so `nvcc` is on PATH (e.g. `module load cuda/12.1`):  bash install.sh
#
set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TV_VERSION="${TV_VERSION:-0.20.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
# gsplat tag built from source. MUST be >= v1.5.0: the model's renderer uses gsplat's
# *batched-scene* rasterization (means [B, N, 3]), which only exists from 1.5.0 on;
# v1.4.0 fails at runtime with `assert means.shape == (N, 3)`.
GSPLAT_REF="${GSPLAT_REF:-v1.5.3}"
# Broad fallback used only if a GPU can't be detected (override with --arch / env).
DEFAULT_ARCH="8.0;8.6;9.0"
# `--arch all` expands to this portable set. Capped at 9.0 because this build uses
# CUDA 12.1, whose nvcc tops out at sm_90 (Blackwell sm_100/sm_120 need CUDA 12.8+).
ALL_ARCHS="7.0;7.5;8.0;8.6;8.9;9.0"

print_help() {
  sed -n '3,31p' "$0" | sed 's/^# \{0,1\}//'
}

expand_arch() {
  case "$1" in
    all|ALL|All) printf '%s' "${ALL_ARCHS}" ;;
    *)           printf '%s' "$1" ;;
  esac
}

# ';'-joined, de-duplicated compute capabilities of the visible GPUs (e.g. "8.0;9.0"),
# read from the driver. Empty if nvidia-smi isn't available / too old to report it.
detect_cuda_arch() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | sed 's/[[:space:]]//g' \
      | grep -E '^[0-9]+\.[0-9]+$' \
      | sort -u \
      | paste -sd';' -
  fi
}

# Fallback that asks torch directly (works once torch is installed and a GPU is visible).
torch_detect_arch() {
  python - <<'PY' 2>/dev/null
import torch
caps = sorted({"%d.%d" % torch.cuda.get_device_capability(i)
               for i in range(torch.cuda.device_count())})
print(";".join(caps))
PY
}

# ---- argument parsing -------------------------------------------------------
ARCH_OVERRIDE=""       # from --arch (highest precedence)
PRINT_ARCH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --arch)   ARCH_OVERRIDE="${2:?--arch needs a value, e.g. --arch \"8.0;9.0\" or --arch all}"; shift 2 ;;
    --arch=*) ARCH_OVERRIDE="${1#--arch=}"; shift ;;
    --print-arch|--detect-arch) PRINT_ARCH=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "unknown option: $1 (try --help)"; exit 2 ;;
  esac
done

# Arch chosen before torch is installed: --arch flag, else env, else driver detect.
choose_arch_pretorch() {
  if [ -n "${ARCH_OVERRIDE}" ]; then
    expand_arch "${ARCH_OVERRIDE}"
  elif [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    printf '%s' "${TORCH_CUDA_ARCH_LIST}"
  else
    detect_cuda_arch || true
  fi
}

# `--print-arch`: report the arch (honouring --arch / env / detection) and exit.
if [ "${PRINT_ARCH}" -eq 1 ]; then
  a="$(choose_arch_pretorch || true)"
  if [ -n "${a}" ]; then
    echo "TORCH_CUDA_ARCH_LIST=${a}"
  else
    echo "Could not detect via nvidia-smi. With torch installed, run:"
    echo "  python -c 'import torch; print(\"%d.%d\" % torch.cuda.get_device_capability())'"
    echo "or pass it explicitly, e.g.:  bash install.sh --arch \"8.0;9.0\"   (or --arch all)"
  fi
  exit 0
fi

# ---- resolve the arch for the build ----------------------------------------
ARCH_SOURCE=""
if [ -n "${ARCH_OVERRIDE}" ]; then
  export TORCH_CUDA_ARCH_LIST="$(expand_arch "${ARCH_OVERRIDE}")"
  ARCH_SOURCE="--arch flag"
elif [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  ARCH_SOURCE="environment"
else
  ARCH="$(detect_cuda_arch || true)"
  if [ -n "${ARCH}" ]; then
    export TORCH_CUDA_ARCH_LIST="${ARCH}"
    ARCH_SOURCE="auto-detected from this node's GPU(s)"
  fi
fi
if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  echo ">> TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (${ARCH_SOURCE})"
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "!! nvcc not found on PATH. gsplat CANNOT compile ahead-of-time and will JIT at"
  echo "!! first run instead. Load your CUDA 12.x toolkit first, e.g.:  module load cuda/12.1"
fi

echo ">> [1/3] torch ${TORCH_VERSION} + torchvision ${TV_VERSION} (CUDA 12.1)"
pip install "torch==${TORCH_VERSION}" "torchvision==${TV_VERSION}" --index-url "${TORCH_INDEX}"

echo ">> [2/3] build tools (needed for the no-isolation, from-source gsplat build)"
# setuptools<81 keeps pkg_resources available for TensorBoard (82 removed it, 81 warns).
pip install ninja "setuptools<81" wheel "numpy==1.26.3"

# Driver couldn't report the arch and the user didn't choose one: ask torch now.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  ARCH="$(torch_detect_arch || true)"
  if [ -n "${ARCH}" ]; then
    export TORCH_CUDA_ARCH_LIST="${ARCH}"
    echo ">> TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (auto-detected via torch)"
  else
    export TORCH_CUDA_ARCH_LIST="${DEFAULT_ARCH}"
    echo "!! Could not detect a GPU; defaulting to TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}."
    echo "!! Targeting specific/multiple devices? Re-run with --arch, e.g.:"
    echo "!!   bash install.sh --arch \"8.0;8.6;9.0\"    (or --arch all)"
  fi
fi

echo ">> [3/3] GlobalSplat (editable) + gsplat ${GSPLAT_REF} compiled from source (archs: ${TORCH_CUDA_ARCH_LIST})"
# Core package first (gsplat is intentionally NOT a core dependency).
pip install -e . --no-build-isolation
# gsplat is pinned HERE (not via the [gpu] extra) so this script is self-contained.
# --no-binary gsplat   -> force a source build so gsplat's CUDA extension (gsplat.csrc)
#                         is compiled NOW, not JIT-compiled at first run.
# --no-build-isolation -> build against the torch installed above (not a fresh cu13x one).
pip install "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_REF}" \
    --no-build-isolation --no-binary gsplat

echo ">> Verifying gsplat kernels are precompiled (no 'Setting up CUDA' message = success):"
python - <<'PY'
import torch, gsplat
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gsplat", gsplat.__version__)
if torch.cuda.is_available():
    caps = sorted({"%d.%d" % torch.cuda.get_device_capability(i)
                   for i in range(torch.cuda.device_count())})
    print("visible GPU arch(s):", ";".join(caps) or "(none)")
try:
    from gsplat.cuda._backend import _C  # noqa: F401
    print("gsplat CUDA backend: PRECOMPILED (no runtime JIT)")
except Exception as e:
    print("gsplat will JIT at first use (csrc not compiled):", e)
PY
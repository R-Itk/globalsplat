# Troubleshooting

Common gsplat / CUDA build and runtime errors and their fixes. See
[INSTALL.md](./INSTALL.md) for the install procedure these refer to.

### `OSError: [Errno 16] Device or resource busy: '.nfsXXXX'` from gsplat (multi-GPU on NFS)

gsplat JIT-compiles its CUDA extension on first use; under DDP, ranks race to
build/clean the same dir on NFS. The training entrypoint already builds it once
on rank 0 behind a barrier (`GsplatWarmupCallback`). For extra safety, build it
once before launching and/or point the cache at node-local scratch:

```bash
export TORCH_EXTENSIONS_DIR=$TMPDIR/torch_ext      # node-local, not NFS
python -c "import torch; from gsplat.cuda._backend import _C"   # one-time build
# then launch the (multi-GPU) run as usual
```

### gsplat prints `Setting up CUDA with MAX_JOBS=...` / `TORCH_CUDA_ARCH_LIST is not set` on the first run

That is gsplat JIT-compiling its kernels at runtime, which means the installed
gsplat has no precompiled `csrc` (a plain `pip install gsplat` ships a JIT build).
Reinstall it from source so the kernels are built at install time:

```bash
# easiest: let install.sh detect (or pass) the arch and rebuild from source
bash install.sh                       # auto-detects this node's arch
bash install.sh --arch "8.0;8.6;9.0"  # or choose explicitly for multiple devices
# equivalently, by hand:
TORCH_CUDA_ARCH_LIST="8.0" pip install -e ".[gpu]" --no-build-isolation --no-binary gsplat
```

Verify with `python -c "from gsplat.cuda._backend import _C"` -- it should return
immediately with no compile message. (If you must keep the JIT build, at least
`export TORCH_CUDA_ARCH_LIST=8.0` and `export TORCH_EXTENSIONS_DIR=$TMPDIR/torch_ext`
so it compiles once to node-local cache; the training entrypoint then serializes
that one-time build across ranks.)

### `RuntimeError: The detected CUDA version (12.x) mismatches the version that was used to compile PyTorch (13.0)` when building gsplat

pip's build isolation pulled a fresh default torch (CUDA 13.0) into a temp env
instead of using your cu121 torch. Build with `--no-build-isolation` (and check
`python -c "import torch; print(torch.version.cuda)"` shows a 12.x build). The
toolkit `nvcc` and torch must share the same CUDA *major* (both 12.x).

### `ModuleNotFoundError` for `dacite` / `matplotlib` / `sk-video` / etc.

These are dependencies of the vendored upstream (ZPressor) code, imported at
runtime. `pip install -e .` pulls them; if you hit one, `pip install <name>`.

### `ModuleNotFoundError: No module named 'pkg_resources'` running `tensorboard`

setuptools 82 removed `pkg_resources`, which TensorBoard imports (81 still works
but warns). Pin `pip install "setuptools<81"` (already pinned in the project deps).

### `add_video needs package moviepy` during validation logging

TensorBoard's `add_video` needs moviepy, and it imports `moviepy.editor` (removed
in moviepy 2.0), so install the 1.x line: `pip install "moviepy<2"`.

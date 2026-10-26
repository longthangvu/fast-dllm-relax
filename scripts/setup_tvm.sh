#!/usr/bin/env bash
# Build the pinned, patched TVM in 3rdparty/tvm.
#
#   bash scripts/setup_tvm.sh cuda     # desktop CUDA workstation (default; also cross-compiles)
#   bash scripts/setup_tvm.sh jetson   # on a Jetson Orin
#   bash scripts/setup_tvm.sh mac      # on an Apple-silicon Mac
#
# Env:
#   CUDA_ARCH   value for CMAKE_CUDA_ARCHITECTURES (e.g. 89, or 87 on Orin). Unset: TVM picks a
#               default list when CUTLASS is on (slower build).
#   JOBS        parallel build jobs (default: all cores)
#   INSTALL_PY=1  also `pip install` the tvm-ffi Python package into the active environment
#               (use a dedicated venv for this repo)
set -euo pipefail

preset="${1:-cuda}"
root="$(cd "$(dirname "$0")/.." && pwd)"
tvm="$root/3rdparty/tvm"
patch="$root/patches/tvm-a7bfc857b.patch"
config="$root/configs/tvm/${preset}.cmake"
[ -f "$config" ] || { echo "unknown preset '$preset' (cuda|jetson|mac)" >&2; exit 1; }

git -C "$root" submodule update --init --recursive

if git -C "$tvm" apply --reverse --check "$patch" 2>/dev/null; then
  echo "patch already applied"
elif git -C "$tvm" apply --check "$patch" 2>/dev/null; then
  git -C "$tvm" apply "$patch"
  echo "patch applied"
else
  # Neither applies: the tree carries another version of the patch (e.g. pulled a newer one).
  # The patch only edits tracked files, so a checkout restores the tree and keeps build/.
  echo "3rdparty/tvm has local changes that conflict with $(basename "$patch")" \
       "(likely an older version of the patch). Reset it and rerun:" >&2
  echo "  git -C $tvm checkout -- . && bash $0 $preset" >&2
  exit 1
fi

mkdir -p "$tvm/build"
{
  cat "$tvm/cmake/config.cmake"
  echo
  echo "# ---- preset: $preset (configs/tvm/${preset}.cmake) ----"
  cat "$config"
  if [ -n "${CUDA_ARCH:-}" ]; then echo "set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})"; fi
} > "$tvm/build/config.cmake"

cd "$tvm/build"
cmake -G Ninja ..
ninja ${JOBS:+-j "$JOBS"}

if [ "${INSTALL_PY:-0}" = "1" ]; then
  pip install "$tvm/3rdparty/tvm-ffi"
fi

echo
echo "TVM built in $tvm/build."
if [ "${INSTALL_PY:-0}" != "1" ]; then
  echo "Install the tvm-ffi Python package once (in this repo's venv):"
  echo "  pip install $tvm/3rdparty/tvm-ffi"
  echo
fi
cat <<MSG
Then, in every shell that uses this repo:
  export TVM_LIBRARY_PATH=$tvm/build
  export PYTHONPATH=$tvm/python:$root\${PYTHONPATH:+:\$PYTHONPATH}
Check: python -c "import tvm; print(tvm.__file__, tvm.support.libinfo()['USE_CUDA'])"
MSG

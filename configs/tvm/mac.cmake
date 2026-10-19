# Mac (Apple silicon, built on the device): runtime for the Metal artifact cross-compiled on the
# workstation with `scripts/export_metal.py`, linked locally with `scripts/link_metal.py`.
# Point USE_LLVM at Homebrew's llvm-config, e.g. /opt/homebrew/opt/llvm/bin/llvm-config.
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_LLVM "/opt/homebrew/opt/llvm/bin/llvm-config --ignore-libllvm --link-static")
set(USE_METAL ON)
set(USE_CUDA OFF)
set(USE_CUBLAS OFF)
set(USE_CUTLASS OFF)

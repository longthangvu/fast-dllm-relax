# Desktop CUDA workstation: compiles every artifact (native CUDA, Jetson cross-compile,
# Metal cross-compile) and runs the native CUDA ones.
# Appended to 3rdparty/tvm/cmake/config.cmake by scripts/setup_tvm.sh; later set() wins.
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(HIDE_PRIVATE_SYMBOLS ON)
# Needs an LLVM with the AArch64 target for Jetson/Mac host code (check: llvm-config --targets-built).
set(USE_LLVM "llvm-config --ignore-libllvm")
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_CUTLASS ON)
# Metal stays OFF here: the patched build_metal_off.cc makes Metal cross-compiled modules
# serializable for a Metal-enabled runtime.
set(USE_METAL OFF)

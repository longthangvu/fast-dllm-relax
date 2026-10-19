# Jetson Orin (built on the device): runtime for artifacts cross-compiled on the workstation
# with `scripts/export_cuda.py --target jetson_orin`. CUTLASS is not used by those artifacts.
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_LLVM OFF)
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_CUTLASS OFF)

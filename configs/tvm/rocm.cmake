# AMD ROCm GPU (Strix Halo gfx1151, Ryzen AI 300/400 gfx1150, ...): builds and runs ROCm artifacts.
# Appended to 3rdparty/tvm/cmake/config.cmake by scripts/setup_tvm.sh; later set() wins.
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(HIDE_PRIVATE_SYMBOLS ON)
# LLVM must know the target GPU (llc -march=amdgcn -mcpu=help | grep gfx1151); Ubuntu llvm-20-dev does.
# ROCm's own /opt/rocm/llvm has no llvm-config, so it cannot be linked here.
set(USE_LLVM "llvm-config --ignore-libllvm")
set(USE_ROCM /opt/rocm)
# hipBLAS BYOC (relax/backend/rocm) is OFF: at a7bfc857b its runtime uses hipblasDatatype_t, which
# hipBLAS 3.x (ROCm 7) removed (use hipDataType), so it does not compile against ROCm 7.2.
set(USE_HIPBLAS OFF)
set(USE_CUDA OFF)
set(USE_CUBLAS OFF)
set(USE_CUTLASS OFF)
set(USE_METAL OFF)

## For LOKI miniforge & VOLTA100 GPUs
current_branch=$(git rev-parse --abbrev-ref HEAD)
build_dir=""

if [ "$current_branch" = "main" ]; then
    build_dir='build-gpu'
elif [ "$current_branch" = "single-cloud-tracking" ]; then
    build_dir='build-gpu-dev'
else
    build_dir="build-$current_branch"
fi

echo "Building dir: $build_dir"

cmake -S . -B build-gpu \
 -DKokkos_ARCH_SKX=ON   -DKokkos_ENABLE_CUDA=ON   -DKokkos_ARCH_AMPERE80=ON   -DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON  \
  -DADIOS2_DIR=$HOME/Packages/adios2-install/lib64/cmake/adios2   -DCMAKE_CXX_STANDARD=17   -DCMAKE_CXX_COMPILER=$(which g++) \
    -DCMAKE_CUDA_COMPILER=$(which nvcc) -DPARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF
cmake --build build-gpu -j 8

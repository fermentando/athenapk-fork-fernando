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

cmake -S. -B"$build_dir" -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON -DHDF5_ROOT=$CONDA_PREFIX  -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python 
cmake --build "$build_dir" -j 8

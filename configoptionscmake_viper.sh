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

cmake -S. -B"$build_dir"   -DKokkos_ARCH_ZEN4=ON  -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX942_APU=ON -DCMAKE_CXX_COMPILER=hipcc -DPARTHENON_DISABLE_HDF5_COMPRESSION=ON -DCMAKE_PREFIX_PATH='/u/ferhi/Packages/adios2-build'
cmake --build "$build_dir" -j 8

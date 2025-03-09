## For LOKI miniforge & VOLTA100 GPUs
current_branch=$(git rev-parse --abbrev-ref HEAD)
build_dir=""

if [ "$current_branch" = "main" ]; then
    build_dir='build-gpu'
elif [ "$current_branch" = "single-cloud-tracking" ]; then
    build_dir='build-gpu-dev'
else
    build_dir="build-new-$current_branch"
fi

echo "Building dir: $build_dir"

cmake -S. -B"$build_dir" -DKokkos_ARCH_ZEN4=ON  -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX942=ON -DCMAKE_CXX_COMPILER=hipcc
cmake --build "$build_dir" -j 8

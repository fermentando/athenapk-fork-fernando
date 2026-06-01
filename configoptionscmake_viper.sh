## For LOKI miniforge & VOLTA100 GPUs
current_branch=$(git rev-parse --abbrev-ref HEAD)
build_dir=""

if [ "$current_branch" = "main" ]; then
    build_dir='build-gpu'
elif [ "$current_branch" = "single-cloud-tracking" ]; then
    build_dir='build-gpu-dev'
elif [ "$current_branch" = "development" ]; then
    build_dir='build-stratified'
else
    build_dir="build-$current_branch"
fi

echo "Building dir: $build_dir"

cmake -S. -B"$build_dir" \
  -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang \
  -DPARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF \
  -DKokkos_ARCH_ZEN4=ON  \
  -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ARCH_AMD_GFX942_APU=ON \
  -DPARTHENON_DISABLE_HDF5_COMPRESSION=ON \
  -DCMAKE_Fortran_COMPILER=gfortran
  #-DCMAKE_INSTALL_PREFIX=$HOME/Packages/kokkos-tools \
  #-DHDF5_ROOT=$HDF5_ROOT 

cmake --build "$build_dir" -j 8

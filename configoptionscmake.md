## Freya NVIDIA VOLTA GPUS

cmake -S. -Bbuild-gpu-v100  -DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON  -DKokkos_ARCH_SKX=ON  -DKokkos_ARCH_VOLTA70=ON -DKokkos_ENABLE_CUDA=ON

## Freya INTEL XEON (G) CPUS with host execution space OPENMP

cmake -S. -Bbuild-cpu  -DKokkos_ENABLE_SERIAL=OFF -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_OPENMP=ON

## CPU serial for Intel Xeon (G)

cmake -S. -Bbuild-serial-cpu -Dkokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_SKX=ON

## Freya NVIDIA PASCAL GPUS

cmake -S. -Bbuild-gpu-p100  -DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON  -DKokkos_ARCH_SKX=ON  -DKokkos_ARCH_PASCAL60=ON -DKokkos_ENABLE_CUDA=ON

## Freya NVIDIA AMPERE GPUS

cmake -S. -Bbuild-gpu-a100  -DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON  -DKokkos_ARCH_SKX=ON  -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_CUDA=ON

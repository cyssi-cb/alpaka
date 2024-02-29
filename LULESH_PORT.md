This Repository is a Alpaka Port of Lulesh (on Cuda GPU) and a original Baseline Version of the Lulesh Benchmark. 

What is Lulesh:
https://asc.llnl.gov/codes/proxy-apps/lulesh

Build Guide for Alpaka:
Follow https://alpaka.readthedocs.io/en/latest/basic/install.html

cmake -Dalpaka_ACC_GPU_CUDA_ENABLE=ON ..

| Alpaka Version of Lulesh | Original Version of Lulesh (using Cmake of Alpaka) |
|----------|----------|
| cmake -Dalpaka_BUILD_ALPAKA_LULESH=ON .. | cmake -Dalpaka_BUILD_LULESH_ORIG=ON .. |
| cmake --build . -t alpaka_lulesh | cmake --build . -t lulesh_orig|
|./alpaka_lulesh/cuda/alpaka_lulesh -s $sqrt3($problemSize) 0 $iterations | ./lulesh_orig/cuda/lulesh_orig -s $sqrt3($problemSize) 0 $iterations |

At the current state only UnfiformCudaHip is permitted as Accelerator for Alpaka.
No MPI support and no support for non-regular Input Parameters (eg. files) which are supported in the Baseline Version.
Input Parameter is the length of an edge on a Cube, where the fluid dynamics equations are performed.


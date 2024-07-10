# octree-miniapp

Demonstration of octree construction on CPUs and GPUs for 3D point clouds.
This is a simplified version of the code found here: [Cornestone Octree](https://github.com/sekelle/cornerstone-octree).
Includes neighbor searches as an example application for the constructed octrees.

## Main features and methods
* Octrees represented based on Space-Filling-Curves (SFCs). Here, we use 3D-Hilbert curves.
* Performance portable octree construction on CPUs and GPUs based on common building blocks such
as radix sort and prefix sums as described in [1].
* Portable neighbor search implementation
* Warp-level optimized neighbor search for the GPU


## Directory structure
```bash
octree-miniapp
├── CMakeLists.txt
├── findneighbors.hpp                - CPU/GPU portable neighbor search implementation
├── findneighbors_warps.cuh          - warp-level optimized neighbor search implementation
├── neighbor_search.cu               - neighbor search mini-app
├── sfc                              - Hilbert SFC implementation
│   ├── bitops.hpp
│   ├── box.hpp
│   └── hilbert.hpp
├── tree                             - octree construction implementation
│   ├── csarray.hpp                  - octree leaf-cell array construction (Sec. 4 of [1])
│   └── octree.hpp                   - internal (fully-linked) octree construction on top of leaf-cells
│                                      (Sec. 5 of [1])
├── util                             - common boiler-plate code
│   ├── accel_switch.hpp
│   ├── annotation.hpp
│   ├── array.hpp
│   ├── cuda_utils.hpp
│   ├── random.hpp
│   ├── stl.hpp
│   ├── timing.cuh
│   └── tuple.hpp
└── warps                            - warp-level functions required for the warp-aware GPU neighbor search
    ├── gpu_config.cuh
    └── warpscan.cuh

```

## Compilation and running the mini-apps
* Major dependencies: Thrust (ships with the CUDA Toolkit)
* HIP support for AMD devices: yes, after hipifying the sources.

```bash
mkdir build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=<60,70 or 80> <THIS_REPO_GIT_SOURCE_DIR>
make -j

# running the mini-apps

```
module load daint-gpu
module load CMake/3.22.1
module swap PrgEnv-cray PrgEnv-gnu
module load cdt/22.05
module swap gcc gcc/11.2.0
module load nvhpc/22.2
```bash

./neighbor_search
```
All executables are single-source, therefore you may also compile them directly on the command line, e.g.:
```bash
nvcc neighbor_search.cu -O3 -std=c++17 -Xcompiler -fopenmp --gpu-architecture=compute_<60,70 or 80> neighbor_search.cu
```
## Paper references
[1] Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations, PASC 2023, [https://doi.org/10.1145/3592979.3593417](https://doi.org/10.1145/3592979.3593417)

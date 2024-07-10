/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Benchmark cornerstone octree generation on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "util/random.hpp"
#include "util/timing.cuh"
#include "sfc/bitops.hpp"
#include "tree/csarray_gpu.cuh"
#include "tree/octree_gpu.cuh"

using namespace cstone;

//! @brief compute SFC keys on the GPU
template<class T, class KeyType>
__global__ void computeSfcKeysGpu(const T* x, const T* y, const T* z, KeyType* keys, int numParticles, const Box<T> box)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles)
    {
        keys[i] = hilbert3D<KeyType>(x[i], y[i], z[i], box);
    }
}

//! @brief compute SFC keys for given particles, then sort keys and rearrange x,y,z in the resulting ordering
template<class T, class KeyType>
void sfcSortParticles(thrust::device_vector<T>& x, thrust::device_vector<T>& y, thrust::device_vector<T>& z,
                      thrust::device_vector<KeyType>& keys, const Box<T>& box)
{
    int numThreads   = 256; 
    int numParticles = x.size();

    computeSfcKeysGpu<<<iceil(numParticles, numThreads), numThreads>>>(rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(keys),
                                                                       numParticles, box);

    //! @brief this will hold the index permutation that sorts the keys
    thrust::device_vector<int> sfcOrder(numParticles);
    thrust::sequence(sfcOrder.begin(), sfcOrder.end(), 0);

    thrust::sort_by_key(keys.begin(), keys.end(), sfcOrder.begin());

    thrust::device_vector<T> temp(numParticles);

    //! @brief reorders the x,y,z arrays into the SFC-sorted ordering
    thrust::gather(sfcOrder.begin(), sfcOrder.end(), x.begin(), temp.begin());
    swap(x, temp);
    thrust::gather(sfcOrder.begin(), sfcOrder.end(), y.begin(), temp.begin());
    swap(y, temp);
    thrust::gather(sfcOrder.begin(), sfcOrder.end(), z.begin(), temp.begin());
    swap(z, temp);
}

int main()
{
    using T = double;
    using KeyType = uint64_t;
    Box<double> box{-1, 1};

    unsigned numParticles = 2000000;
    unsigned bucketSize   = 16;

    RandomCoordinates<T, KeyType> randomBox(numParticles, box);

    thrust::device_vector<T> d_x = randomBox.x();
    thrust::device_vector<T> d_y = randomBox.y();
    thrust::device_vector<T> d_z = randomBox.z();

    thrust::device_vector<KeyType> particleKeys(numParticles);

    sfcSortParticles(d_x, d_y, d_z, particleKeys, box);

    thrust::device_vector<KeyType>  octree = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
    thrust::device_vector<unsigned> counts = std::vector<unsigned>{numParticles};

    thrust::device_vector<KeyType>       tmpTree;
    thrust::device_vector<TreeNodeIndex> workArray;

    auto fullBuild = [&]()
    {
        while (!updateOctreeGpu(rawPtr(particleKeys), rawPtr(particleKeys) + numParticles, bucketSize, octree, counts,
                                tmpTree, workArray))
            ;
    };

    float buildTime = timeGpu(fullBuild);
    std::cout << "build time from scratch " << buildTime / 1000 << " nNodes(tree): " << nNodes(octree)
              << " particle count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;

    auto updateTree = [&]()
    {
        updateOctreeGpu(rawPtr(particleKeys), rawPtr(particleKeys) + numParticles, bucketSize, octree, counts, tmpTree,
                        workArray);
    };

    float updateTime = timeGpu(updateTree);
    std::cout << "build time with guess " << updateTime / 1000 << " nNodes(tree): " << nNodes(octree)
              << " particle count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;

    OctreeData<KeyType, GpuTag> linkedOctree;
    linkedOctree.resize(nNodes(octree));
    auto buildInternal = [&]() { buildLinkedTreeGpu(rawPtr(octree), linkedOctree.data()); };

    float internalBuildTime = timeGpu(buildInternal);
    std::cout << "linked octree build time " << internalBuildTime / 1000 << std::endl << std::endl;

    thrust::host_vector<TreeNodeIndex> levelRange = linkedOctree.levelRange; // download from GPU
    for (int i = 0; i < levelRange.size(); ++i)
    {
        int numNodes = levelRange[i + 1] - levelRange[i];
        if (numNodes == 0) { break; }
        std::cout << "number of nodes at level " << i << ": " << numNodes << std::endl;
    }
}

/*! @file
 * @brief  Find neighbors in Space-Filling-Curve sorted x,y,z arrays
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include <thrust/device_vector.h>

#include "util/cuda_utils.hpp"
#include "util/random.hpp"
#include "util/timing.cuh"

#include "findneighbors.hpp"
#include "findneighbors_warps.cuh"

// uncomment to enable warp-level optimized neighbor search
// #define USE_WARPS

using namespace cstone;

template<class T, class KeyType>
__global__ void findNeighborsKernel(const T* x, const T* y, const T* z, const T* h, LocalIndex numParticles,
                                    const Box<T> box, const OctreeNsView<T, KeyType> treeView, unsigned ngmax,
                                    LocalIndex* neighbors, unsigned* neighborsCount)
{
   // TODO: implement me

   // launch neighbor search for all particles in parallel
}

template<class T, class KeyType>
void benchmarkGpu(int numParticles, bool verbose)
{
    Box<T> box{0, 1, BoundaryType::open};
    int    maxNeighbors = 200;

    /****** Particle data and tree generation ****************/
    RandomCoordinates<T, KeyType> coords(numParticles, box);
    std::vector<T>                h(numParticles, 0.012);

    const T*       x    = coords.x().data();
    const T*       y    = coords.y().data();
    const T*       z    = coords.z().data();
    const KeyType* keys = coords.keys().data();

    unsigned bucketSize   = 64; // maximum number of particles per leaf node
    auto [csTree, counts] = computeOctree(keys, keys + numParticles, bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    buildLinkedTree<KeyType>(csTree.data(), octree.data());
    const TreeNodeIndex* childOffsets = octree.childOffsets.data();
    const TreeNodeIndex* toLeafOrder  = octree.internalToLeaf.data();

    std::vector<LocalIndex> layout(nNodes(csTree) + 1);
    std::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), 0);

    std::vector<Vec3<T>> nodeCenters(octree.numNodes), nodeSizes(octree.numNodes);
    nodeFpCenters(octree.prefixes.data(), octree.numNodes, nodeCenters.data(), nodeSizes.data(), box);

    OctreeNsView<T, KeyType> treeView{nodeCenters.data(), nodeSizes.data(), octree.childOffsets.data(),
                                      octree.internalToLeaf.data(), layout.data()};

    /****** CPU output data ****************/
    std::vector<LocalIndex> neighborsCPU(maxNeighbors * numParticles);
    std::vector<unsigned>   neighborsCountCPU(numParticles);

    auto findNeighborsCpu = [&]()
    {
#pragma omp parallel for
        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            neighborsCountCPU[i] = findNeighbors(i, x, y, z, h.data(), treeView, box, maxNeighbors,
                                                 neighborsCPU.data() + i * maxNeighbors);
        }
    };

    float cpuTime = timeCpu(findNeighborsCpu);

    std::cout << "CPU time " << cpuTime << " s" << std::endl;
    if (verbose)
    {
        std::copy(neighborsCountCPU.data(), neighborsCountCPU.data() + 64, std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
    }

    /****** Verification: compare against all-2-all ****************/
    bool all2allpass = true;
    if (numParticles <= 10000)
    {
        std::vector<unsigned> neighborsCountRef(numParticles);
        findNeighborsAll2All(x, y, z, h.data(), numParticles, neighborsCountRef.data());

        all2allpass = std::equal(neighborsCountCPU.begin(), neighborsCountCPU.end(), neighborsCountRef.begin());
        std::cout << "CPU all-2-all reference: " << (all2allpass ? "PASS" : "FAIL") << std::endl;
    }
    else { std::cout << "CPU all-2-all reference: SKIPPED (numParticles > 10000)" << std::endl; }

    /****** Upload input data to GPU ****************/
    thrust::device_vector<T> d_x(coords.x().begin(), coords.x().end());
    thrust::device_vector<T> d_y(coords.y().begin(), coords.y().end());
    thrust::device_vector<T> d_z(coords.z().begin(), coords.z().end());
    thrust::device_vector<T> d_h = h;

    thrust::device_vector<Vec3<T>>       d_nodeCenters    = nodeCenters;
    thrust::device_vector<Vec3<T>>       d_nodeSizes      = nodeSizes;
    thrust::device_vector<TreeNodeIndex> d_childOffsets   = octree.childOffsets;
    thrust::device_vector<TreeNodeIndex> d_internalToLeaf = octree.internalToLeaf;
    thrust::device_vector<LocalIndex>    d_layout         = layout;

    OctreeNsView<T, KeyType> treeViewGpu{rawPtr(d_nodeCenters), rawPtr(d_nodeSizes), rawPtr(d_childOffsets),
                                         rawPtr(d_internalToLeaf), rawPtr(d_layout)};

    /****** GPU output data ****************/
    thrust::device_vector<LocalIndex> d_neighbors(numParticles * maxNeighbors);
    thrust::device_vector<unsigned>   d_neighborsCount(numParticles);

    auto findNeighborsLambda = [&]()
    {
#ifdef USE_WARPS
        // the fast warp-aware version
        findNeighborsBT(0, numParticles, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), treeViewGpu, box,
                        rawPtr(d_neighborsCount), rawPtr(d_neighbors), maxNeighbors);
#else
        findNeighborsKernel<<<iceil(numParticles, 128), 128>>>(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h),
                                                               numParticles, box, treeViewGpu, maxNeighbors,
                                                               rawPtr(d_neighbors), rawPtr(d_neighborsCount));
#endif
    };

    float gpuTime = timeGpu(findNeighborsLambda);
    std::cout << "GPU time " << gpuTime / 1000 << " s" << std::endl;

    /****** Download GPU results ****************/

    std::vector<cstone::LocalIndex> neighborsGPU(maxNeighbors * numParticles);
    std::vector<unsigned>           neighborsCountGPU(numParticles);

    thrust::copy(d_neighborsCount.begin(), d_neighborsCount.end(), neighborsCountGPU.begin());
    thrust::copy(d_neighbors.begin(), d_neighbors.end(), neighborsGPU.begin());

    /****** Verification: compare against CPU ****************/

    if (verbose)
    {
        std::copy(neighborsCountGPU.data(), neighborsCountGPU.data() + 64, std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
    }

    int numFails     = 0;
    int numFailsList = 0;
    for (int i = 0; i < numParticles; ++i)
    {
        std::sort(neighborsCPU.data() + i * maxNeighbors,
                  neighborsCPU.data() + i * maxNeighbors + neighborsCountCPU[i]);

        std::vector<cstone::LocalIndex> nilist(neighborsCountGPU[i]);
        for (unsigned j = 0; j < neighborsCountGPU[i]; ++j)
        {
#ifdef USE_WARPS
            // access pattern for the warp-aware version
            size_t warpOffset = (i / TravConfig::targetSize) * TravConfig::targetSize * maxNeighbors;
            size_t laneOffset = i % TravConfig::targetSize;
            nilist[j]         = neighborsGPU[warpOffset + TravConfig::targetSize * j + laneOffset];
            nilist[j]         = neighborsGPU[warpOffset + TravConfig::targetSize * j + laneOffset];
#else
            nilist[j] = neighborsGPU[i * maxNeighbors + j];
#endif
        }
        std::sort(nilist.begin(), nilist.end());

        if (neighborsCountGPU[i] != neighborsCountCPU[i])
        {
            if (verbose) std::cout << i << " " << neighborsCountGPU[i] << " " << neighborsCountCPU[i] << std::endl;
            numFails++;
        }

        // Also check if the neighbors are the same, not just the number of neighbors
        if (!std::equal(begin(nilist), end(nilist), neighborsCPU.begin() + i * maxNeighbors)) { numFailsList++; }
    }

    bool allEqual = std::equal(begin(neighborsCountGPU), end(neighborsCountGPU), begin(neighborsCountCPU));
    if (allEqual && all2allpass)
        std::cout << "GPU neighbor counts: PASS\n";
    else
        std::cout << "GPU neighbor counts: FAIL " << numFails << std::endl;

    std::cout << "numFailsList " << numFailsList << std::endl;
}

int main(int argc, char** argv)
{
    int  numParticles = (argc > 1) ? std::stoi(argv[1]) : 10000;
    bool verbose      = false;

    std::cout << "Performing neighbor search for " << numParticles << " particles." << std::endl;
    benchmarkGpu<double, uint64_t>(numParticles, verbose);
}

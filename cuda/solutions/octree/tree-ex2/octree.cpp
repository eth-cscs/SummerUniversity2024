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
 * @brief Octree performance test
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include <numeric>

#include "util/random.hpp"
#include "tree/octree.hpp"
#include "util/timing.cuh"

using namespace cstone;

template<class T, class KeyType>
void sfcSortParticles(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, std::vector<KeyType>& keys,
                      const Box<T>& box)
{
    std::size_t n = x.size();
    computeSfcKeys(x.data(), y.data(), z.data(), keys.data(), n, box);

    std::vector<std::size_t> sfcOrder(n);
    std::iota(begin(sfcOrder), end(sfcOrder), std::size_t(0));
    sort_by_key(begin(keys), end(keys), begin(sfcOrder));

    std::vector<T> temp(x.size());
    gather(sfcOrder.data(), sfcOrder.size(), x.data(), temp.data());
    swap(x, temp);
    gather(sfcOrder.data(), sfcOrder.size(), y.data(), temp.data());
    swap(y, temp);
    gather(sfcOrder.data(), sfcOrder.size(), z.data(), temp.data());
    swap(z, temp);
}

int main()
{
    using KeyType = uint64_t;
    Box<double> box{-1, 1};

    unsigned numParticles = 2000000;
    unsigned bucketSize   = 16;

    RandomCoordinates<double, KeyType> coords(numParticles, box);
    std::vector<KeyType>               keys(numParticles);

    sfcSortParticles(coords.x(), coords.y(), coords.z(), keys, box);

    // initialize with the root node containing all particles
    std::vector<KeyType> octree{0, nodeRange<KeyType>(0)};
    // note brace initializer: this vector has length 1
    std::vector<unsigned> counts{numParticles};

    auto fullBuild = [&]()
    {
        while (!updateOctree(keys.data(), keys.data() + numParticles, bucketSize, octree, counts))
            ;
    };

    float buildTime = timeCpu(fullBuild);
    std::cout << "build time from scratch " << buildTime << " nNodes(tree): " << nNodes(octree)
              << " particle count: " << std::accumulate(begin(counts), end(counts), 0lu) << std::endl;

    auto updateTree = [&]()
    { updateOctree(keys.data(), keys.data() + numParticles, bucketSize, octree, counts); };

    float updateTime    = timeCpu(updateTree);
    long  numEmptyNodes = std::count(begin(counts), end(counts), 0);

    std::cout << "build time with guess " << updateTime << " nNodes(tree): " << nNodes(octree)
              << " particle count: " << std::accumulate(begin(counts), end(counts), 0lu)
              << " empty nodes: " << numEmptyNodes << std::endl;

    OctreeData<KeyType, CpuTag> linkedOctree;
    linkedOctree.resize(nNodes(octree));
    auto buildInternal = [&]() { buildLinkedTree<KeyType>(rawPtr(octree), linkedOctree.data()); };

    float internalBuildTime = timeCpu(buildInternal);
    std::cout << "linked octree build time " << internalBuildTime << std::endl << std::endl;

    for (int i = 0; i < linkedOctree.levelRange.size(); ++i)
    {
        int numNodes = linkedOctree.levelRange[i + 1] - linkedOctree.levelRange[i];
        if (numNodes == 0) { break; }
        std::cout << "number of nodes at level " << i << ": " << numNodes << std::endl;
    }
}

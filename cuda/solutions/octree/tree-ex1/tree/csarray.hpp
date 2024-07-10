/*! @file
 * @brief Generation of local and global octrees in cornerstone format
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * In the cornerstone format, the octree is stored as sequence of SFC codes
 * fulfilling three invariants. Each code in the sequence both signifies the
 * the start SFC code of an octree leaf node and serves as an upper SFC code bound
 * for the previous node.
 *
 * The invariants of the cornerstone format are:
 *      - code sequence contains code 0 and the maximum code 2^30 or 2^61
 *      - code sequence is sorted by ascending code value
 *      - difference between consecutive elements must be a power of 8
 *
 * The consequences of these invariants are:
 *      - the entire space is covered, i.e. there are no holes in the tree
 *      - only leaf nodes are stored
 *      - for each leaf node, all its siblings (nodes at the same subdivision level with
 *        the same parent) are present in the SFC code sequence
 *      - each node with index i is defined by its lowest possible SFC code at position
 *        i in the vector and the highest possible (excluding) SFC code at position i+1
 *        in the vector
 *      - a vector of length N represents N-1 leaf nodes
 */

#pragma once

#include <numeric>
#include <vector>
#include <tuple>

#include "../util/stl.hpp"
#include "../util/tuple.hpp"

namespace cstone
{

//! @brief Controls the node index type, has to be signed.
using TreeNodeIndex = int;
//! @brief index type of local particle arrays
using LocalIndex = unsigned;

/*! @brief returns the number of nodes in a tree leaf histogram
 *
 * @tparam    Vector  a vector-like container that has a .size() member
 * @param[in] tree    input tree histogram
 * @return            the number of leaf nodes
 *
 * This makes it explicit that a vector of n SFC keys corresponds to a tree with n - 1 nodes,
 * or a histogram with n elements has n - 1 buckets.
 */
template<class Vector>
std::size_t nNodes(const Vector& tree)
{
    assert(tree.size());
    return tree.size() - 1;
}

//! @brief count particles in one tree node
template<class KeyType>
HOST_DEVICE_FUN unsigned calculateNodeCount(
    KeyType nodeStart, KeyType nodeEnd, const KeyType* codesStart, const KeyType* codesEnd, size_t maxCount)
{
    // count particles in range
    auto rangeStart = stl::lower_bound(codesStart, codesEnd, nodeStart);
    auto rangeEnd   = stl::lower_bound(codesStart, codesEnd, nodeEnd);
    size_t count    = rangeEnd - rangeStart;

    return stl::min(count, maxCount);
}

/*! @brief count number of particles in each octree node
 *
 * @tparam       KeyType      32- or 64-bit unsigned integer type
 * @param[in]    tree         octree nodes given as SFC codes of length @a nNodes+1
 *                            needs to satisfy the octree invariants
 * @param[inout] counts       output particle counts per node, length = @a nNodes
 * @param[in]    numNodes     number of nodes in tree
 * @param[in]    codesStart   sorted particle SFC code range start
 * @param[in]    codesEnd     sorted particle SFC code range end
 * @param[in]    maxCount     maximum particle count per node to store
 */
template<class KeyType>
void computeNodeCounts(const KeyType* tree,
                       unsigned* counts,
                       TreeNodeIndex numNodes,
                       const KeyType *codesStart,
                       const KeyType *codesEnd,
                       unsigned maxCount)
{
#pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        counts[i] = calculateNodeCount(tree[i], tree[i + 1], codesStart, codesEnd, maxCount);
    }
}

/*! @brief return the sibling index and level of the specified csTree node
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param  csTree    cornerstone octree, length N
 * @param  nodeIdx   node index in [0:N] of @p csTree to compute sibling index
 * @return           in first pair element: index in [0:8] if all 8 siblings of the specified
 *                   node are next to each other and at the same division level.
 *                   -1 otherwise, i.e. if not all the 8 siblings exist in @p csTree
 *                   at the same division level
 *                   in second pair element: tree level of node at @p nodeIdx
 *
 * Sibling nodes are group of 8 leaf nodes that have the same parent node.
 */
template<class KeyType>
inline HOST_DEVICE_FUN util::tuple<int, unsigned> siblingAndLevel(const KeyType* csTree, TreeNodeIndex nodeIdx)
{
    KeyType thisNode = csTree[nodeIdx];
    KeyType range    = csTree[nodeIdx + 1] - thisNode;
    unsigned level   = treeLevel(range);

    if (level == 0) { return {-1, level}; }

    int siblingIdx = octalDigit(thisNode, level);
    bool siblings  = (csTree[nodeIdx - siblingIdx + 8] == csTree[nodeIdx - siblingIdx] + nodeRange<KeyType>(level - 1));
    if (!siblings) { siblingIdx = -1; }

    return {siblingIdx, level};
}

//! @brief returns 0 for merging, 1 for no-change, 8 for splitting
template<class KeyType>
HOST_DEVICE_FUN int
calculateNodeOp(const KeyType* tree, TreeNodeIndex nodeIdx, const unsigned* counts, unsigned bucketSize)
{
    auto [siblingIdx, level] = siblingAndLevel(tree, nodeIdx);

    if (siblingIdx > 0) // 8 siblings next to each other, node can potentially be merged
    {
        // pointer to first node in sibling group
        auto g             = counts + nodeIdx - siblingIdx;
        size_t parentCount = size_t(g[0]) + size_t(g[1]) + size_t(g[2]) + size_t(g[3]) + size_t(g[4]) + size_t(g[5]) +
                             size_t(g[6]) + size_t(g[7]);
        bool countMerge = parentCount <= size_t(bucketSize);
        if (countMerge) { return 0; } // merge
    }

    if (counts[nodeIdx] > bucketSize && level < maxTreeLevel<KeyType>{}) { return 8; } // split

    return 1; // default: do nothing
}

/*! @brief Compute split or fuse decision for each octree node in parallel
 *
 * @tparam    KeyType      32- or 64-bit unsigned integer type
 * @param[in] tree         octree nodes given as SFC codes of length @p nNodes
 *                         needs to satisfy the octree invariants
 * @param[in] counts       output particle counts per node, length = @p nNodes
 * @param[in] nNodes       number of nodes in tree
 * @param[in] bucketSize   maximum particle count per (leaf) node and
 *                         minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps     stores rebalance decision result for each node, length = @p nNodes
 * @return                 true if all nodes are unchanged, false otherwise
 *
 * For each node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class KeyType, class LocalIndex>
bool rebalanceDecision(
    const KeyType* tree, const unsigned* counts, TreeNodeIndex nNodes, unsigned bucketSize, LocalIndex* nodeOps)
{
    bool converged = true;

#pragma omp parallel
    {
        bool convergedThread = true;
#pragma omp for
        for (TreeNodeIndex i = 0; i < nNodes; ++i)
        {
            int decision = calculateNodeOp(tree, i, counts, bucketSize);
            if (decision != 1) { convergedThread = false; }

            nodeOps[i] = decision;
        }
        if (!convergedThread) { converged = false; }
    }
    return converged;
}

/*! @brief transform old nodes into new nodes based on opcodes
 *
 * @tparam KeyType    32- or 64-bit integer
 * @param  nodeIndex  the node to process in @p oldTree
 * @param  oldTree    the old tree
 * @param  nodeOps    opcodes for each old tree node
 * @param  newTree    the new tree
 */
template<class KeyType>
HOST_DEVICE_FUN void
processNode(TreeNodeIndex nodeIndex, const KeyType* oldTree, const TreeNodeIndex* nodeOps, KeyType* newTree)
{
    KeyType thisNode = oldTree[nodeIndex];
    KeyType range    = oldTree[nodeIndex + 1] - thisNode;
    unsigned level   = treeLevel(range);

    TreeNodeIndex opCode       = nodeOps[nodeIndex + 1] - nodeOps[nodeIndex];
    TreeNodeIndex newNodeIndex = nodeOps[nodeIndex];

    if (opCode == 1) { newTree[newNodeIndex] = thisNode; }
    else if (opCode == 8)
    {
        for (int sibling = 0; sibling < 8; ++sibling)
        {
            // insert new nodes
            newTree[newNodeIndex + sibling] = thisNode + sibling * nodeRange<KeyType>(level + 1);
        }
    }
}

/*! @brief split or fuse octree nodes based on node counts relative to bucketSize
 *
 * @param[in]    tree         cornerstone octree
 * @param[out]   newTree      rebalanced cornerstone octree
 * @param[in]    nodeOps      rebalance decision for each node, length @p numNodes(tree) + 1
 *                            will be overwritten
 */
template<class InputVector, class OutputVector>
void rebalanceTree(const InputVector& tree, OutputVector& newTree, TreeNodeIndex* nodeOps)
{
    TreeNodeIndex numNodes = nNodes(tree);

    std::exclusive_scan(nodeOps, nodeOps + numNodes + 1, nodeOps, 0); // add 1 to store the total sum
    TreeNodeIndex newNumNodes = nodeOps[numNodes];

    newTree.resize(newNumNodes + 1); // histogram size is number of nodes + 1

#pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        processNode(i, tree.data(), nodeOps, newTree.data());
    }
    newTree.back() = tree.back();
}

/*! @brief update the octree with a single rebalance/count step
 *
 * @tparam       KeyType     32- or 64-bit unsigned integer for SFC code
 * @param[in]    firstKey    first local particle SFC key
 * @param[in]    lastKey     last local particle SFC key
 * @param[in]    bucketSize  maximum number of particles per node
 * @param[inout] tree        the octree leaf nodes (cornerstone format)
 * @param[inout] counts      the octree leaf node particle count
 * @param[in]    maxCount    if actual node counts are higher, they will be capped to @p maxCount
 * @return                   true if tree was not modified, false otherwise
 */
template<class KeyType>
bool updateOctree(const KeyType* firstKey,
                  const KeyType* lastKey,
                  unsigned bucketSize,
                  std::vector<KeyType>& tree,
                  std::vector<unsigned>& counts,
                  unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    std::vector<TreeNodeIndex> nodeOps(nNodes(tree) + 1);
    bool converged = rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

    std::vector<KeyType> tmpTree;
    rebalanceTree(tree, tmpTree, nodeOps.data());
    swap(tree, tmpTree);

    counts.resize(nNodes(tree));
    computeNodeCounts(tree.data(), counts.data(), nNodes(tree), firstKey, lastKey, maxCount);

    return converged;
}

//! @brief Convenience wrapper for updateOctree. Start from scratch and return a fully converged cornerstone tree.
template<class KeyType>
std::tuple<std::vector<KeyType>, std::vector<unsigned>>
computeOctree(const KeyType* codesStart,
              const KeyType* codesEnd,
              unsigned bucketSize,
              unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    std::vector<KeyType> tree{0, nodeRange<KeyType>(0)};
    std::vector<unsigned> counts{unsigned(codesEnd - codesStart)};

    while (!updateOctree(codesStart, codesEnd, bucketSize, tree, counts, maxCount))
        ;

    return std::make_tuple(std::move(tree), std::move(counts));
}

} // namespace cstone

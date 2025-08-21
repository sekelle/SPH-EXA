/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  MPI extension for calculating distributed cornerstone octrees
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <chrono>
#include <mpi.h>
#include <span>

#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/tree/csarray_gpu.h"
#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/update_mpi.hpp"
#include "cstone/util/pack_buffers.hpp"

namespace cstone
{

//! @brief sum reduction like MPI_SUM, but cap maximum values before they overflow
inline void sumCapped(void* inP, void* inoutP, int* len, MPI_Datatype*)
{
    auto* in    = reinterpret_cast<unsigned*>(inP);
    auto* inout = reinterpret_cast<unsigned*>(inoutP);
    for (int i = 0; i < *len; ++i)
    {
        auto a   = in[i];
        auto b   = inout[i];
        inout[i] = a + b >= std::max(a, b) ? a + b : std::numeric_limits<unsigned>::max();
    }
}

/*! @brief global update step of an octree, including regeneration of the internal node structure
 *
 * @tparam        KeyType     unsigned 32- or 64-bit integer
 * @param[in]     keys        first particle key, on device
 * @param[in]     bucketSize  max number of particles per leaf
 * @param[out]    tree        output fully linked octree built on top of updated leaves
 * @param[inout]  d_csTree    leaf nodes
 * @param[out]    d_countsBuf leaf node particle counts
 * @param[in]     expectOverflows  use sum-reduction that guards against integer overflow if true
 * @return                         maximum number of particles per cell capped to 2^32-1 or 0 if tree has max depth
 */
template<class KeyType, class DevKeyVec, class DevCountVec>
unsigned updateOctreeGlobalGpu(std::span<const KeyType> keys,
                               unsigned bucketSize,
                               OctreeData<KeyType, GpuTag>& tree,
                               DevKeyVec& d_csTree,
                               DevCountVec& d_countsBuf,
                               bool expectOverflows,
                               std::span<float> timing = {})
{
    auto t0 = std::chrono::high_resolution_clock::now();
    auto newNumNodes =
        computeNodeOpsGpu(d_csTree.data(), nNodes(d_csTree), d_countsBuf.data(), bucketSize, tree.childOffsets.data());
    reallocate(tree.prefixes, newNumNodes + 1, 1.01);
    bool converged = rebalanceTreeGpu(d_csTree.data(), nNodes(d_csTree), newNumNodes, tree.childOffsets.data(),
                                      tree.prefixes.data());
    swap(d_csTree, tree.prefixes);

    tree.resize(newNumNodes);
    buildOctreeGpu(d_csTree.data(), tree.data());

    size_t numLeafNodes = tree.numLeafNodes;
    auto [d_counts, d_countsRed] =
        util::packAllocBuffer(d_countsBuf, util::TypeList<unsigned, unsigned>{}, {numLeafNodes, numLeafNodes}, 2 << 20);

    computeNodeCountsGpu(rawPtr(d_csTree), d_counts.data(), numLeafNodes, keys, std::numeric_limits<unsigned>::max(),
                         true);

    syncGpu();
    std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
    if (expectOverflows)
    {
        MPI_Op limitSum;
        MPI_Op_create(&sumCapped, true, &limitSum);
        mpiAllreduceGpuDirect(d_counts.data(), d_countsRed.data(), d_counts.size(), limitSum, MPI_COMM_WORLD);
        MPI_Op_free(&limitSum);
    }
    else
    {
        t1 = std::chrono::high_resolution_clock::now();
        mpiAllreduceGpuDirect(d_counts.data(), d_countsRed.data(), d_counts.size(), MPI_SUM, MPI_COMM_WORLD);
        t2 = std::chrono::high_resolution_clock::now();
    }
    sequenceMax(d_counts.data(), d_counts.data() + d_counts.size(), d_countsRed.data(), d_counts.data());
    d_countsBuf.resize(numLeafNodes);
    auto t3 = std::chrono::high_resolution_clock::now();

    if (timing.size() >= 3)
    {
        timing[0] = std::chrono::duration<float>(t1 - t0).count(); // assign::globalUpdate
        timing[1] = std::chrono::duration<float>(t2 - t1).count(); // assign::allreduce
        timing[2] = std::chrono::duration<float>(t3 - t2).count(); // assign::seqDl
    }

    if (converged) { return 0; }

    auto [minCount, maxCount] = MinMaxGpu<unsigned>{}(d_counts.data(), d_counts.data() + d_counts.size());
    return maxCount;
}

template<class KeyType, class Accelerator, class DevKeyVec, class DevCountVec>
unsigned updateOctreeGlobal(std::span<const KeyType> keys,
                            unsigned bucketSize,
                            OctreeData<KeyType, Accelerator>& tree,
                            std::vector<KeyType>& leaves,
                            DevKeyVec& d_csTree,
                            std::vector<unsigned>& counts,
                            DevCountVec& d_counts,
                            bool firstCall,
                            std::span<float> timing = {})
{
    if constexpr (HaveGpu<Accelerator>{})
    {
        return updateOctreeGlobalGpu(keys, bucketSize, tree, d_csTree, d_counts, firstCall, timing);
    }
    else { return updateOctreeGlobal(keys, bucketSize, tree, leaves, counts); }
}

} // namespace cstone

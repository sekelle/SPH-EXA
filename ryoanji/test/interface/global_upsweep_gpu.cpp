/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Compute an octree and multipoles on GPUs from a set of particles distributed across ranks
 *        and compare against a single-node reference computed from the same set.
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>

#define USE_CUDA
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/domain/domain.hpp"
#include "cstone/findneighbors.hpp"
#include "cstone/tree/csarray.hpp"
#include "coord_samples/random.hpp"


#include "ryoanji/interface/global_multipole.hpp"
#include "ryoanji/interface/multipole_holder.cuh"

using namespace ryoanji;

template<class T, class KeyType>
static int multipoleHolderTest(int thisRank, int numRanks)
{
    using MultipoleType              = CartesianQuadrupole<T>;
    const LocalIndex numParticles    = 1000 * numRanks;
    unsigned         bucketSize      = 64;
    unsigned         bucketSizeLocal = 16;
    float            theta           = 1.0;

    cstone::Box<T> box{-1, 1};

    // common pool of coordinates, identical on all ranks
    cstone::RandomGaussianCoordinates<T, cstone::SfcKind<KeyType>> coords(numRanks * numParticles, box);

    std::vector<T> globalH(numRanks * numParticles, 0.1);
    adjustSmoothingLength<KeyType>(globalH.size(), 5, 10, coords.x(), coords.y(), coords.z(), globalH, box);

    std::vector<T> globalMasses(numRanks * numParticles, 1.0 / (numRanks * numParticles));

    auto firstIndex = numParticles * thisRank;
    auto lastIndex  = numParticles * thisRank + numParticles;

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T> x(coords.x().begin() + firstIndex, coords.x().begin() + lastIndex);
    std::vector<T> y(coords.y().begin() + firstIndex, coords.y().begin() + lastIndex);
    std::vector<T> z(coords.z().begin() + firstIndex, coords.z().begin() + lastIndex);
    std::vector<T> h(globalH.begin() + firstIndex, globalH.begin() + lastIndex);
    std::vector<T> m(globalMasses.begin() + firstIndex, globalMasses.begin() + lastIndex);

    std::vector<KeyType> particleKeys(x.size());

    cstone::Domain<KeyType, T, cstone::GpuTag> domain(thisRank, numRanks, bucketSize, bucketSizeLocal, theta, box);

    MultipoleHolder<T, T, T, T, T, KeyType, MultipoleType> multipoleHolder;

    cstone::DeviceVector<KeyType> d_keys = particleKeys;
    cstone::DeviceVector<T>       d_x = x, d_y = y, d_z = z, d_h = h, d_m = m;
    cstone::DeviceVector<T>       s1, s2, s3;
    domain.syncGrav(d_keys, d_x, d_y, d_z, d_h, d_m, std::tuple{}, std::tie(s1, s2, s3));
    domain.exchangeHalos(std::tie(d_m), s1, s2);

    //! includes tree plus associated information, like peer ranks, assignment, counts, centers, etc
    const cstone::FocusedOctree<KeyType, T, cstone::GpuTag>& focusTree = domain.focusTree();
    //! the focused octree, structure only
    auto octree = focusTree.octreeViewAcc();

    multipoleHolder.upsweep(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), domain.globalTree(), domain.focusTree(),
                            domain.layout().data());

    // Check the root multipole of the distributed tree
    std::array<int, 2> testResults{0, 0};
    {
        // globally replicated tree built from reference particle set with full LET resolution everywhere
        auto [refLeaves, refCounts] = cstone::computeOctree<KeyType>(coords.particleKeys(), bucketSizeLocal);
        cstone::OctreeData<KeyType, cstone::CpuTag> refOctree;
        refOctree.resize(cstone::nNodes(refLeaves));
        cstone::updateInternalTree<KeyType>(refLeaves, refOctree.data());

        std::vector<LocalIndex> refLayout(refOctree.numLeafNodes + 1);
        std::inclusive_scan(refCounts.begin(), refCounts.end(), refLayout.begin() + 1);

        std::vector<cstone::SourceCenterType<T>> refCenters(refOctree.numNodes);

#pragma omp parallel for schedule(static)
        for (TreeNodeIndex leafIdx = 0; leafIdx < refOctree.numLeafNodes; ++leafIdx)
        {
            TreeNodeIndex nodeIdx = refOctree.leafToInternal[refOctree.numInternalNodes + leafIdx];
            refCenters[nodeIdx] =
                cstone::massCenter<T>(coords.x().data(), coords.y().data(), coords.z().data(), globalMasses.data(),
                                      refLayout[leafIdx], refLayout[leafIdx + 1]);
        }
        cstone::upsweep(refOctree.levelRange, refOctree.childOffsets.data(), refCenters.data(),
                        cstone::CombineSourceCenter<T>{});
        cstone::setMac<T, KeyType>(refOctree.prefixes, refCenters, 1.0 / theta, box);

        std::vector<MultipoleType> multipoles(octree.numNodes);
        memcpyD2H(multipoleHolder.deviceMultipoles(), multipoles.size(), multipoles.data());

        auto                                     d_centers = focusTree.expansionCentersAcc();
        std::vector<cstone::SourceCenterType<T>> centers(d_centers.size());
        memcpyD2H(d_centers.data(), d_centers.size(), centers.data());

        std::vector<KeyType> letKeys(octree.numNodes);
        memcpyD2H(octree.prefixes, octree.numNodes, letKeys.data());

        int numCentersFail = 0;
        for (TreeNodeIndex i = 0; i < octree.numNodes; ++i)
        {
            auto refIdx = cstone::locateNode(letKeys[i], refOctree.prefixes.data(), refOctree.levelRange.data());
            for (std::size_t c = 0; c < centers[i].size(); ++c)
            {
                if (std::abs(centers[i][c] - refCenters[refIdx][c]) > 1e-6) { numCentersFail++; }
            }
        }

        // compute reference root cell multipole from global particle data
        MultipoleType reference;
        P2M(coords.x().data(), coords.y().data(), coords.z().data(), globalMasses.data(), 0, numParticles * numRanks,
            centers[0], reference);

        MultipoleType globalRootMultipole = multipoles[0];
        double maxDiff = max(abs(reference - globalRootMultipole));

        bool pass      = maxDiff < 1e-10;
        testResults[0] = pass;
        testResults[1] = numCentersFail == 0;
        mpiAllreduce(MPI_IN_PLACE, testResults.data(), testResults.size(), MPI_SUM, MPI_COMM_WORLD);
    }

    bool testPassed = testResults[0] == numRanks && testResults[1] == numRanks;
    if (thisRank == 0)
    {
        std::string r1 = testResults[0] == numRanks ? "PASS" : "FAIL";
        std::cout << "Upsweep test result: " << r1 << std::endl;
        std::string r2 = testResults[1] == numRanks ? "PASS" : "FAIL";
        std::cout << "Center-of-mass test result: " << r1 << std::endl;
    }

    if (testPassed) { return EXIT_SUCCESS; }
    else { return EXIT_FAILURE; }
}

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int testResult = multipoleHolderTest<double, uint64_t>(rank, numRanks);

    MPI_Finalize();

    return testResult;
}

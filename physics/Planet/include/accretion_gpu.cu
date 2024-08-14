//
// Created by Noah Kubli on 12.03.2024.
//
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "sph/util/device_math.cuh"

#include "cstone/sfc/box.hpp"
#include "cstone/tree/definitions.h"

#include "sph/particles_data.hpp"

#include "accretion_gpu.hpp"
#include "star_data.hpp"
#include "cuda_runtime.h"

static __device__ double   dev_accr_mass;
static __device__ double   dev_accr_mom_x;
static __device__ double   dev_accr_mom_y;
static __device__ double   dev_accr_mom_z;
static __device__ unsigned dev_n_removed;
static __device__ unsigned dev_n_accreted;

using cstone::TravConfig;

template<typename T1, typename Th, typename Tremove, typename T2, typename Tm, typename Tv>
__global__ void computeAccretionConditionKernel(size_t first, size_t last, const T1* x, const T1* y, const T1* z,
                                                const Th* h, Tremove* remove, const Tm* m, const Tv* vx, const Tv* vy,
                                                const Tv* vz, T2 star_x, T2 star_y, T2 star_z, T2 star_size2,
                                                T2 removal_limit_h)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    double             accr_mass{};
    double             accr_mom_x{};
    double             accr_mom_y{};
    double             accr_mom_z{};
    unsigned           accreted{};
    unsigned           removed{};

    if (i >= last) {}
    else
    {
        const double dx    = x[i] - star_x;
        const double dy    = y[i] - star_y;
        const double dz    = z[i] - star_z;
        const double dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 < star_size2)
        {
            // Accrete on star
            remove[i]  = cstone::removeKey<Tremove>::value;
            accr_mass  = m[i];
            accr_mom_x = m[i] * vx[i];
            accr_mom_y = m[i] * vy[i];
            accr_mom_z = m[i] * vz[i];
            accreted   = 1;
        }
        else if (h[i] > removal_limit_h)
        {
            // Remove from system
            remove[i] = cstone::removeKey<Tremove>::value;
            removed   = 1;
        }
    }
    typedef cub::BlockReduce<double, TravConfig::numThreads> BlockReduceDouble;

    __shared__ typename BlockReduceDouble::TempStorage temp_accr_mass;

    double block_accr_mass = BlockReduceDouble(temp_accr_mass).Reduce(accr_mass, cub::Sum());

    __shared__ typename BlockReduceDouble::TempStorage temp_accr_mom_x, temp_accr_mom_y, temp_accr_mom_z;

    double block_accr_mom_x = BlockReduceDouble(temp_accr_mom_x).Reduce(accr_mom_x, cub::Sum());
    double block_accr_mom_y = BlockReduceDouble(temp_accr_mom_y).Reduce(accr_mom_y, cub::Sum());
    double block_accr_mom_z = BlockReduceDouble(temp_accr_mom_z).Reduce(accr_mom_z, cub::Sum());

    typedef cub::BlockReduce<unsigned, TravConfig::numThreads> BlockReduceUnsigned;

    __shared__ typename BlockReduceUnsigned::TempStorage temp_storage_n_rem, temp_storage_n_accr;

    unsigned block_n_removed  = BlockReduceUnsigned(temp_storage_n_rem).Reduce(removed, cub::Sum());
    unsigned block_n_accreted = BlockReduceUnsigned(temp_storage_n_accr).Reduce(accreted, cub::Sum());

    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(&dev_accr_mass, block_accr_mass);
        atomicAdd(&dev_accr_mom_x, block_accr_mom_x);
        atomicAdd(&dev_accr_mom_y, block_accr_mom_y);
        atomicAdd(&dev_accr_mom_z, block_accr_mom_z);
        atomicAdd(&dev_n_removed, block_n_removed);
        atomicAdd(&dev_n_accreted, block_n_accreted);
    }
}

template<typename Dataset, typename StarData>
void computeAccretionConditionGPU(size_t first, size_t last, Dataset& d, StarData& star)
{
    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    double   zero   = 0.;
    unsigned zero_s = 0;
    cudaMemcpyToSymbol(dev_accr_mass, &zero, sizeof(zero));
    cudaMemcpyToSymbol(dev_accr_mom_x, &zero, sizeof(zero));
    cudaMemcpyToSymbol(dev_accr_mom_y, &zero, sizeof(zero));
    cudaMemcpyToSymbol(dev_accr_mom_z, &zero, sizeof(zero));
    cudaMemcpyToSymbol(dev_n_removed, &zero_s, sizeof(zero_s));
    cudaMemcpyToSymbol(dev_n_accreted, &zero_s, sizeof(zero_s));

    computeAccretionConditionKernel<<<numBlocks, numThreads>>>(
        first, last, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.h),
        rawPtr(d.devData.keys), rawPtr(d.devData.m), rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz),
        star.position[0], star.position[1], star.position[2], star.inner_size * star.inner_size, star.removal_limit_h);
    checkGpuErrors(cudaGetLastError());
    checkGpuErrors(cudaDeviceSynchronize());

    double   m_accr_ret;
    double   px_accr_ret;
    double   py_accr_ret;
    double   pz_accr_ret;
    unsigned n_removed;
    unsigned n_accr;

    cudaMemcpyFromSymbol(&m_accr_ret, dev_accr_mass, sizeof(m_accr_ret));
    cudaMemcpyFromSymbol(&px_accr_ret, dev_accr_mom_x, sizeof(px_accr_ret));
    cudaMemcpyFromSymbol(&py_accr_ret, dev_accr_mom_y, sizeof(py_accr_ret));
    cudaMemcpyFromSymbol(&pz_accr_ret, dev_accr_mom_z, sizeof(pz_accr_ret));
    cudaMemcpyFromSymbol(&n_removed, dev_n_removed, sizeof(n_removed));
    cudaMemcpyFromSymbol(&n_accr, dev_n_accreted, sizeof(n_accr));

    star.m_accreted_local    = m_accr_ret;
    star.p_accreted_local[0] = px_accr_ret;
    star.p_accreted_local[1] = py_accr_ret;
    star.p_accreted_local[2] = pz_accr_ret;
    star.n_removed_local     = n_removed;
    star.n_accreted_local    = n_accr;
}

template void computeAccretionConditionGPU(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>&, StarData&);

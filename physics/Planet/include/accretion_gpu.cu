//
// Created by Noah Kubli on 12.03.2024.
//
#include <cub/cub.cuh>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "sph/util/device_math.cuh"

#include "cstone/sfc/box.hpp"

#include "accretion_gpu.hpp"
#include "cuda_runtime.h"

template<typename T1, typename Tremove, typename T2>
__global__ void computeAccretionConditionKernel(size_t first, size_t last, const T1* x, const T1* y, const T1* z,
                                                Tremove* remove, T2 star_x, T2 star_y, T2 star_z, T2 star_size2)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }

    const double dx    = x[i] - star_x;
    const double dy    = y[i] - star_y;
    const double dz    = z[i] - star_z;
    const double dist2 = dx * dx + dy * dy + dz * dz;

    if (dist2 < star_size2) remove[i] = 1;
}

template<typename T1, typename Tremove, typename T2>
void computeAccretionConditionGPU(size_t first, size_t last, const T1* x, const T1* y, const T1* z, Tremove* remove,
                               const T2* spos, T2 star_size)
{
    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    computeAccretionConditionKernel<<<numBlocks, numThreads>>>(first, last, x, y, z, remove, spos[0], spos[1], spos[2],
                                                               star_size * star_size);
    checkGpuErrors(cudaGetLastError());
}

template void computeAccretionConditionGPU(size_t, size_t, const double*, const double*, const double*, uint64_t*,
                                        const double*, double);
struct is_zero
{
    __device__ bool operator()(const uint64_t& k) { return (k == 0); }
};

template<typename T1, typename Tremove>
void moveAccretedToEndGPU(size_t first, size_t last, T1* x, Tremove* remove, size_t* n_removed)
{

    const auto* part_ptr = thrust::stable_partition(thrust::device, x + first, x + last, remove + first, is_zero{});
    *n_removed           = (x + last) - part_ptr;
}

template void moveAccretedToEndGPU(size_t, size_t, double*, uint64_t*, size_t*);
template void moveAccretedToEndGPU(size_t, size_t, float*, uint64_t*, size_t*);

template<typename Tv, typename Tm, typename Tstar>
void sumMassAndMomentumGPU(size_t sum_first, size_t sum_last, const Tv* vx, const Tv* vy, const Tv* vz, const Tm* m,
                           Tstar* m_sum, Tstar* p_sum)
{
    if (sum_first == sum_last)
    {
        *m_sum   = 0.;
        p_sum[0] = 0.;
        p_sum[1] = 0.;
        p_sum[2] = 0.;
    }
    thrust::device_vector<Tv> px(sum_last - sum_first);
    thrust::device_vector<Tv> py(sum_last - sum_first);
    thrust::device_vector<Tv> pz(sum_last - sum_first);

    thrust::copy(thrust::device, vx + sum_first, vx + sum_last, px.begin());
    thrust::copy(thrust::device, vy + sum_first, vy + sum_last, py.begin());
    thrust::copy(thrust::device, vz + sum_first, vz + sum_last, pz.begin());

    thrust::transform(px.begin(), px.end(), thrust::device_ptr<const Tm>(m) + sum_first, px.begin(),
                      thrust::multiplies<Tv>{});
    thrust::transform(py.begin(), py.end(), thrust::device_ptr<const Tm>(m) + sum_first, py.begin(),
                      thrust::multiplies<Tv>{});
    thrust::transform(pz.begin(), pz.end(), thrust::device_ptr<const Tm>(m) + sum_first, pz.begin(),
                      thrust::multiplies<Tv>{});

    p_sum[0] = thrust::reduce(px.begin(), px.end(), 0., thrust::plus<Tv>{});
    p_sum[1] = thrust::reduce(py.begin(), py.end(), 0., thrust::plus<Tv>{});
    p_sum[2] = thrust::reduce(pz.begin(), pz.end(), 0., thrust::plus<Tv>{});

    *m_sum = thrust::reduce(thrust::device_ptr<const Tm>(m) + sum_first, thrust::device_ptr<const Tm>(m) + sum_last, 0.,
                            thrust::plus<Tm>{});
}

template void sumMassAndMomentumGPU(size_t, size_t, const float*, const float*, const float*, const float*, double*,
                                    double*);

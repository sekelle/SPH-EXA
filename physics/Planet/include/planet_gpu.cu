//
// Created by Noah Kubli on 11.03.2024.
//
#include <cub/cub.cuh>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "sph/util/device_math.cuh"

#include "cstone/sfc/box.hpp"

#include "planet_gpu.hpp"
#include "cuda_runtime.h"

//__device__ static double star_force_device[3];
//__device__ static double star_potential_device;

template<typename T1, typename Ta, typename Tm, typename T2>
__global__ void computeCentralForceGPUKernel(size_t first, size_t last, const T1* x, const T1* y, const T1* z, Ta* dax,
                                             Ta* day, Ta* daz, const Tm* m, T2 sx, T2 sy, T2 sz, T2 sm, T1 G,
                                             double* star_force_block0, double* star_force_block1,
                                             double* star_force_block2, double* star_potential_block)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }

    double star_force_thread[3]  = {};
    double star_potential_thread = 0.;

    const double dx    = x[i] - sx;
    const double dy    = y[i] - sy;
    const double dz    = z[i] - sz;
    const double dist2 = dx * dx + dy * dy + dz * dz;
    const double dist  = sqrt(dist2);
    const double dist3 = dist2 * dist;

    // Assume stellar mass is 1 and G = 1.
    const double a_strength = 1. / dist3 * sm * G;
    const double ax         = -dx * a_strength;
    const double ay         = -dy * a_strength;
    const double az         = -dz * a_strength;
    dax[i] += ax;
    day[i] += ay;
    daz[i] += az;

    star_force_thread[0] -= ax * m[i];
    star_force_thread[1] -= ay * m[i];
    star_force_thread[2] -= az * m[i];
    star_potential_thread -= G * m[i] / dist;

    typedef cub::BlockReduce<double, cstone::TravConfig::numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage                     temp_storage0;
    __shared__ typename BlockReduce::TempStorage                     temp_storage1;
    __shared__ typename BlockReduce::TempStorage                     temp_storage2;
    __shared__ typename BlockReduce::TempStorage                     temp_storage3;

    BlockReduce reduce0(temp_storage0);
    BlockReduce reduce1(temp_storage1);
    BlockReduce reduce2(temp_storage2);
    BlockReduce reduce3(temp_storage3);

    double star_force_reduced0    = reduce0.Reduce(star_force_thread[0], cub::Sum());
    double star_force_reduced1    = reduce1.Reduce(star_force_thread[1], cub::Sum());
    double star_force_reduced2    = reduce2.Reduce(star_force_thread[2], cub::Sum());
    double star_potential_reduced = reduce3.Reduce(star_potential_thread, cub::Sum());

    // star_force_block0[blockIdx.x]
    //     star_force_block1[blockIdx.x]
    //     star_force_block2[blockIdx.x]
    //     star_potential_block[blockIdx.x]
    //

    // double           star_force_block0 = reduce0.Reduce(star_force_thread[0], cub::Sum());
    // double           star_force_block1 = reduce1.Reduce(star_force_thread[1], cub::Sum());
    // double           star_force_block2 = reduce2.Reduce(star_force_thread[2], cub::Sum());
    // double           star_potential_block = reduce3.Reduce(star_potential_thread, cub::Sum());

    __syncthreads();

    if (threadIdx.x == 0)
    {
        star_force_block0[blockIdx.x]    = star_force_reduced0;
        star_force_block1[blockIdx.x]    = star_force_reduced1;
        star_force_block2[blockIdx.x]    = star_force_reduced2;
        star_potential_block[blockIdx.x] = star_potential_reduced;

        /*if (blockIdx.x == 0)
        {
            for (size_t i = 1; i < blockDim.x; i++)
            {

                star_force_block0[0] += star_force_block0[i];
                star_force_block1[0] += star_force_block1[i];
                star_force_block2[0] += star_force_block2[i];
                star_potential_block[0] += star_potential_block[i];
            }
        }*/

        // atomicSum(&star_force_device[0], star_force_block0);
        /*atomicSum(&star_force_device[1], star_force_block1);
        atomicSum(&star_force_device[2], star_force_block2);
        atomicSum(&star_potential_device, star_potential_block);*/
    }
}

template<typename T1, typename Ta, typename Tm, typename T2>
void computeCentralForceGPU(size_t first, size_t last, const T1* x, const T1* y, const T1* z, Ta* ax, Ta* ay, Ta* az,
                            const Tm* m, const T2* spos, T2 sm, T2* sf, T2* spot, T1 G)
{
    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    // Set to 0
    sf[0] = 0.;
    sf[1] = 0.;
    sf[2] = 0.;
    *spot = 0.;
    // checkGpuErrors(cudaMemcpyToSymbol(star_force_device, sf, sizeof *sf * 3));
    // checkGpuErrors(cudaMemcpyToSymbol(star_potential_device, sf, sizeof *spot));

    T2* star_force_block0;
    cudaMalloc(&star_force_block0, sizeof(T2) * numBlocks);
    T2* star_force_block1;
    cudaMalloc(&star_force_block1, sizeof(T2) * numBlocks);
    T2* star_force_block2;
    cudaMalloc(&star_force_block2, sizeof(T2) * numBlocks);
    T2* star_potential_block;
    cudaMalloc(&star_potential_block, sizeof(T2) * numBlocks);

    computeCentralForceGPUKernel<<<numBlocks, numThreads>>>(first, last, x, y, z, ax, ay, az, m, spos[0], spos[1],
                                                            spos[2], sm, G, star_force_block0, star_force_block1,
                                                            star_force_block2, star_potential_block);
    checkGpuErrors(cudaGetLastError());
    // checkGpuErrors(cudaMemcpyFromSymbol(sf, star_force_device, (sizeof *sf) * 3));
    // checkGpuErrors(cudaMemcpyFromSymbol(spot, &star_potential_device, sizeof *spot));

    sf[0] = thrust::reduce(thrust::device_ptr<T2>(star_force_block0),
                           thrust::device_ptr<T2>(star_force_block0) + numBlocks, 0., thrust::plus<T2>{});
    sf[1] = thrust::reduce(thrust::device_ptr<T2>(star_force_block1),
                           thrust::device_ptr<T2>(star_force_block1) + numBlocks, 0., thrust::plus<T2>{});
    sf[2] = thrust::reduce(thrust::device_ptr<T2>(star_force_block2),
                           thrust::device_ptr<T2>(star_force_block2) + numBlocks, 0., thrust::plus<T2>{});
    *spot = thrust::reduce(thrust::device_ptr<T2>(star_potential_block),
                           thrust::device_ptr<T2>(star_potential_block) + numBlocks, 0., thrust::plus<T2>{});

    // checkGpuErrors(cudaMemcpy(sf, star_force_block0, (sizeof *sf), cudaMemcpyDeviceToHost));
    // checkGpuErrors(cudaMemcpy(sf + 1, star_force_block1, (sizeof *sf), cudaMemcpyDeviceToHost));
    // checkGpuErrors(cudaMemcpy(sf + 2, star_force_block2, (sizeof *sf), cudaMemcpyDeviceToHost));
    // checkGpuErrors(cudaMemcpy(spot, star_potential_block, (sizeof *spot), cudaMemcpyDeviceToHost));

    // checkGpuErrors(cudaMemcpyFromSymbol(sf, star_force_block0, (sizeof *sf)));
    // checkGpuErrors(cudaMemcpyFromSymbol(sf + 1, star_force_block0, (sizeof *sf)));
    // checkGpuErrors(cudaMemcpyFromSymbol(sf + 2, star_force_block0, (sizeof *sf)));
    // checkGpuErrors(cudaMemcpyFromSymbol(spot, star_potential_block, sizeof *spot));

    cudaFree(star_force_block0);
    cudaFree(star_force_block1);
    cudaFree(star_force_block2);
    cudaFree(star_potential_block);
}

template void computeCentralForceGPU(size_t, size_t, const double*, const double*, const double*, float*, float*,
                                     float*, const float*, const double*, double, double*, double*, double);

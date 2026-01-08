/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Cornerstone octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include "cstone/cuda/cub.hpp"

#include "gtest/gtest.h"

#include "cstone/focus/source_center_gpu.h"

using namespace cstone;

template<class T>
struct NormMass
{
    HOST_DEVICE_FUN SourceCenterType<T> operator()(const SourceCenterType<T>& p)
    {
        T invM = p[3] != T(0.0) ? T(1) / p[3] : T(1);
        return Vec4<T>{p[0] * invM, p[1] * invM, p[2] * invM, p[3]};
    }
};

template<class T>
struct Square
{
    HOST_DEVICE_FUN T operator()(T x) { return x * x; }
};

TEST(FocusGpu, SourceCenter)
{
    using Tc = float;
    using Tf = float;

    thrust::device_vector<Tc> x = {-1,-2,1,2,1};
    thrust::device_vector<Tc> y = {-2,-4,1,2,1};
    thrust::device_vector<Tc> z = {-4,-6,1,2,1};
    thrust::device_vector<Tc> m = {1,1,1,1,1};
    thrust::device_vector<LocalIndex> segments = {0, 2, 5};

    thrust::device_vector<TreeNodeIndex> leafToInternal{1, 4};
    thrust::device_vector<SourceCenterType<Tf>> centers(10);

    auto rawPtr = [](auto& vec) { return thrust::raw_pointer_cast(vec.data()); };

    computeLeafSourceCenterGpuNew(rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(m), rawPtr(leafToInternal), segments.size() - 1, rawPtr(segments), rawPtr(centers));

    std::cout << "centers[1] " << static_cast<Vec4<Tf>>(centers[1])[0] //
                      << " " << static_cast<Vec4<Tf>>(centers[1])[1] //
                      << " " << static_cast<Vec4<Tf>>(centers[1])[2] //
                      << " " << static_cast<Vec4<Tf>>(centers[1])[3] << std::endl;

    std::cout << "centers[4] " << static_cast<Vec4<Tf>>(centers[4])[0] //
                        << " " << static_cast<Vec4<Tf>>(centers[4])[1] //
                        << " " << static_cast<Vec4<Tf>>(centers[4])[2] //
                        << " " << static_cast<Vec4<Tf>>(centers[4])[3] << std::endl;

    if (false)
    {
        auto xperm   = thrust::make_permutation_iterator(x.begin(), leafToInternal.begin());
        auto xpermsq = thrust::transform_output_iterator(xperm, Square<Tf>{});

        xpermsq[0] = 4;
        //Tf xp0 = xpermsq[0];
        std::cout << x[1] << std::endl;
    }
}

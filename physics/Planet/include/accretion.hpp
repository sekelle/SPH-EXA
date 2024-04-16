//
// Created by Noah Kubli on 15.03.2024.
//

#pragma once

#include <cassert>
#include <mpi.h>
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/fields/field_get.hpp"
#include "cstone/tree/accel_switch.hpp"
#include "cstone/cuda/cuda_stubs.h"
#include "cstone/util/type_list.hpp"

#include "sph/particles_data.hpp"

#include "accretion_impl.hpp"
#include "accretion_gpu.hpp"
#include "fieldListExclude.hpp"

namespace planet
{

template<typename Dataset, typename StarData>
void computeAccretionCondition(size_t first, size_t last, Dataset& d, const StarData& star)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeAccretionConditionGPU(first, last, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z),
                                     rawPtr(d.devData.keys), star.position.data(), star.inner_size);
    }
    else
    {
        computeAccretionConditionImpl(first, last, d.x.data(), d.y.data(), d.z.data(), d.keys.data(),
                                      star.position.data(), star.inner_size);
    }
}

template<util::StructuralString exclude, typename Fields>
using FieldListExclude_t = FieldListExclude<exclude, Fields>::value;

//! @brief Keys should not be in conserved fields.
template<typename ConservedFields, typename DependentFields, typename Dataset, typename StarData>
void computeNewOrder(size_t first, size_t last, Dataset& d, StarData& star)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeNewOrderGPU(first, last, rawPtr(get<"keys">(d)), &star.n_accreted);
    }
    else { computeNewOrderImpl(first, last, d.keys.data(), &star.n_accreted); }
}

//! @brief Keys should not be in conserved fields.
template<typename ConservedFields, typename DependentFields, typename Dataset, typename StarData>
void applyNewOrder(size_t first, size_t last, Dataset& d, StarData& star)
{
    using SortFields = decltype(util::FieldList<"x", "y", "z", "h">{} + ConservedFields{});
    auto sortVectors = get<SortFields>(d);

    using ScratchFields       = FieldListExclude_t<"keys", DependentFields>;
    auto scratch_fields_tuple = [&d]()
    {
        if constexpr (std::tuple_size_v<decltype(util::make_array(ScratchFields{}))> == 1)
        {
            return std::make_tuple(get<ScratchFields>(d));
        }
        else { return get<ScratchFields>(d); }
    };
    auto scratchVectors = scratch_fields_tuple();

    util::for_each_tuple(
        [&](auto& vector)
        {
            auto& scratch = util::pickType<std::decay_t<decltype(vector)>&>(scratchVectors);
            if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
            {
                applyNewOrderGPU(first, last, rawPtr(vector), rawPtr(scratch), rawPtr(get<"keys">(d)));
            }
            else { applyNewOrderImpl(first, last, vector.data(), scratch.data(), d.keys.data()); }
        },
        sortVectors);
}

template<typename Dataset, typename StarData>
void sumAccretedMassAndMomentum(size_t first, size_t last, Dataset& d, StarData& star)
{
    if (star.n_accreted > (last - first)) throw std::runtime_error("Accreting more particles than on rank");
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        sumMassAndMomentumGPU(last - star.n_accreted, last, rawPtr(get<"vx">(d)), rawPtr(get<"vy">(d)),
                              rawPtr(get<"vz">(d)), rawPtr(get<"m">(d)), &star.m_accreted_local,
                              star.p_accreted_local.data());
    }
    else
    {
        sumMassAndMomentumImpl(last - star.n_accreted, last, get<"vx">(d).data(), get<"vy">(d).data(),
                               get<"vz">(d).data(), get<"m">(d).data(), &star.m_accreted_local,
                               star.p_accreted_local.data());
    }
}

template<typename StarData>
void exchangeAndAccreteOnStar(StarData& star, double minDt_m1, int rank)
{
    double                m_accreted_global{};
    std::array<double, 3> p_accreted_global{};

    MPI_Reduce(&star.m_accreted_local, &m_accreted_global, 1, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(star.p_accreted_local.data(), p_accreted_global.data(), 3, MpiType<double>{}, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::array<double, 3> momentum_star_old;
        std::array<double, 3> momentum_star_new;
        momentum_star_old[0] = (star.position_m1[0] / minDt_m1) * star.m;
        momentum_star_old[1] = (star.position_m1[1] / minDt_m1) * star.m;
        momentum_star_old[2] = (star.position_m1[2] / minDt_m1) * star.m;
        momentum_star_new[0] = momentum_star_old[0] + p_accreted_global[0];
        momentum_star_new[1] = momentum_star_old[1] + p_accreted_global[1];
        momentum_star_new[2] = momentum_star_old[2] + p_accreted_global[2];

        double m_new = m_accreted_global + star.m;

        star.position_m1[0] = momentum_star_new[0] / m_new * minDt_m1;
        star.position_m1[1] = momentum_star_new[1] / m_new * minDt_m1;
        star.position_m1[2] = momentum_star_new[2] / m_new * minDt_m1;

        star.m = m_new;
    }
    if (rank == 0) { printf("accreted mass: %g\tstar mass: %lf\n", m_accreted_global, star.m); }

    MPI_Bcast(star.position_m1.data(), 3, MpiType<double>{}, 0, MPI_COMM_WORLD);
    MPI_Bcast(&star.m, 1, MpiType<double>{}, 0, MPI_COMM_WORLD);
}

}; // namespace planet

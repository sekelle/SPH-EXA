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

#include "sph/particles_data.hpp"

#include "accretion_impl.hpp"
#include "accretion_gpu.hpp"

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

template<typename ActiveFields, typename Dataset, typename StarData>
void moveAccretedToEnd(size_t first, size_t last, Dataset& d, StarData& star)
{
    auto fields = get<ActiveFields>(d);
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        util::for_each_tuple(
            [&](auto& field)
            { moveAccretedToEndGPU(first, last, rawPtr(field), rawPtr(d.devData.keys), &star.n_accreted); },
            fields);
    }
    else
    {
        util::for_each_tuple([&](auto& field)
                             { moveAccretedToEndImpl(first, last, field.data(), d.keys.data(), &star.n_accreted); },
                             fields);
    }
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

        double m_new        = m_accreted_global + star.m;
        star.position_m1[0] = momentum_star_new[0] / m_new * minDt_m1;
        star.position_m1[1] = momentum_star_new[1] / m_new * minDt_m1;
        star.position_m1[2] = momentum_star_new[2] / m_new * minDt_m1;

        star.m = m_new;
    }
    if (rank == 0) { printf("accreted mass: %g\tstar mass: %lf\n", m_accreted_global, star.m); }

    MPI_Bcast(star.position_m1.data(), 3, MpiType<double>{}, 0, MPI_COMM_WORLD);
    MPI_Bcast(&star.m, 1, MpiType<double>{}, 0, MPI_COMM_WORLD);
}

/*template<typename ActiveFields, typename Dataset, typename Domain, typename StarData>
void accreteParticlesGPU(Dataset& d, Domain& domain, StarData& star, double r_outer = 25.)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const size_t first = domain.startIndex();
    const size_t last  = domain.endIndex();

    computeAccretionCondition(first, last, d, star);

    moveAccretedToEnd<ActiveFields>(first, last, d, star);

    sumAccretedMassAndMomentum(first, last, d, star);

    // Send to ranks
    exchangeAndAccreteOnStar(star, d.minDt_m1, rank);

    domain.setEndIndex(last - star.n_accreted);
}*/

// template<typename ActiveFields, typename Dataset, typename Domain, typename StarData>
// void accreteParticles(Dataset& d, Domain& domain, StarData& star, double r_outer = 25.)
//{
//     // if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
//     // {
//     accreteParticlesGPU<ActiveFields>(d, domain, star, r_outer);
//     // }
//     /*else
//     {
//         auto copyContent = [&d](size_t from, size_t to)
//         {
//             if (from == to) return;
//             auto fields_from = cstone::getPointers(get<ActiveFields>(d), from);
//             auto fields_to   = cstone::getPointers(get<ActiveFields>(d), to);
//             for_each_tuples([]<typename T>(const T* from_ptr, T* to_ptr) { *to_ptr = *from_ptr; }, fields_from,
//                             fields_to);
//         };
//
//         size_t endIndexNew     = domain.endIndex();
//         auto   delete_particle = [&endIndexNew, &domain, copyContent](size_t i)
//         {
//             assert(endIndexNew > 0);
//             assert(endIndexNew - domain.startIndex > 0);
//             assert(i > startIndex);
//
//             copyContent(endIndexNew - 1, i);
//             endIndexNew--;
//         };
//         // Go through particles, identify which to delete.
//         double dm_local     = 0.;
//         double dmom_x_local = 0.;
//         double dmom_y_local = 0.;
//         double dmom_z_local = 0.;
//         for (size_t i = domain.startIndex(); i < domain.endIndex(); i++)
//         {
//             const double x     = d.x[i] - star.position[0];
//             const double y     = d.y[i] - star.position[1];
//             const double z     = d.z[i] - star.position[2];
//             const double dist2 = x * x + y * y + z * z;
//             if (dist2 < star.inner_size * star.inner_size || dist2 > r_outer * r_outer)
//             {
//                 // star.m += d.m[i];
//                 dm_local += d.m[i];
//                 dmom_x_local += d.vx[i] * d.m[i];
//                 dmom_y_local += d.vy[i] * d.m[i];
//                 dmom_z_local += d.vz[i] * d.m[i];
//
//                 // double momentum_x = d.vx[i] * d.m[i];
//                 // double momentum_y = d.vy[i] * d.m[i];
//                 // double momentum_z = d.vz[i] * d.m[i];
//
//                 // star.position_m1[0] += momentum_x / star.m; mal Zeit
//                 // star.position_m1[1] += momentum_y / star.m;
//                 // star.position_m1[2] += momentum_z / star.m;
//
//                 delete_particle(i);
//             }
//         }
//         printf("removed: %zu\n", domain.endIndex() - endIndexNew);
//         domain.setEndIndex(endIndexNew);
//         // If delete, swap with endIndex - 1, decrease endIndex, save newEndIndex.
//
//         double dm_global{};
//         double dmom_x_global{};
//         double dmom_y_global{};
//         double dmom_z_global{};
//
//         MPI_Reduce(&dm_local, &dm_global, 1, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
//         MPI_Reduce(&dmom_x_local, &dmom_x_global, 1, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
//         MPI_Reduce(&dmom_y_local, &dmom_y_global, 1, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
//         MPI_Reduce(&dmom_z_local, &dmom_z_global, 1, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
//
//         int rank = 0;
//         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//         if (rank == 0)
//         {
//             star.m += dm_global;
//             star.position_m1[0] += dmom_x_local / star.m * d.minDt_m1;
//             star.position_m1[1] += dmom_y_local / star.m * d.minDt_m1;
//             star.position_m1[2] += dmom_z_local / star.m * d.minDt_m1;
//         }
//         MPI_Bcast(star.position_m1.data(), 3, MpiType<double>{}, 0, MPI_COMM_WORLD);
//         MPI_Bcast(&star.m, 1, MpiType<double>{}, 0, MPI_COMM_WORLD);
//         // Get momentum and mass of particle and add to star.
//         // Synchronize star position
//     }*/
// }
}; // namespace planet
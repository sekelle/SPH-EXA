//
// Created by Noah Kubli on 04.03.2024.
//

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/fields/field_get.hpp"
#include "cstone/tree/accel_switch.hpp"
#include "cstone/cuda/cuda_stubs.h"

#include "planet_gpu.hpp"

namespace planet
{

template<typename Dataset, typename StarData>
void computeCentralForce(Dataset& d, size_t startIndex, size_t endIndex, StarData& star)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeCentralForceGPU(startIndex, endIndex, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z),
                               rawPtr(d.devData.ax), rawPtr(d.devData.ay), rawPtr(d.devData.az), rawPtr(d.devData.m),
                               star.position.data(), star.m, star.force_local.data(), &star.potential_local, d.g);
    }
    else
    {
        // std::array<double, 3> f_tot{};
        star.force_local = {0., 0., 0.};
        // Potential energy coming from interaction with star
        // double pot_star = 0.;
        star.potential_local = 0.;

        for (size_t i = startIndex; i < endIndex; i++)
        {
            const double x     = d.x[i] - star.position[0];
            const double y     = d.y[i] - star.position[1];
            const double z     = d.z[i] - star.position[2];
            const double dist2 = x * x + y * y + z * z;
            const double dist  = std::sqrt(dist2);
            const double dist3 = dist2 * dist;

            // Assume stellar mass is 1 and G = 1.
            const double a_strength = 1. / dist3 * star.m * d.g;
            const double ax         = -x * a_strength;
            const double ay         = -y * a_strength;
            const double az         = -z * a_strength;
            d.ax[i] += ax;
            d.ay[i] += ay;
            d.az[i] += az;

            star.force_local[0] -= ax * d.m[i];
            star.force_local[1] -= ay * d.m[i];
            star.force_local[2] -= az * d.m[i];
            star.potential_local -= d.g * d.m[i] / dist;
        }
        // return std::make_tuple(pot_star, f_tot);
    }
}

template<typename StarData>
void computeAndExchangeStarPosition(StarData& star, double dt, double dt_m1)
{
    std::array<double, 3> global_force{};

    MPI_Reduce(star.force_local.data(), global_force.data(), 3, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&star.potential_local, &star.potential, 1, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // printf("rank: %d total_force: %lf\t%lf\t%lf\n", rank, total_force[0],total_force[1],total_force[2]);

    // printf("rank: %d global_force: %lf\t%lf\t%lf\n", rank, global_force[0],global_force[1],global_force[2]);

    if (rank == 0)
    {
        // a = f / M; M == 1
        double a_starx = global_force[0] / star.m;
        double a_stary = global_force[1] / star.m;
        double a_starz = global_force[2] / star.m;

        auto integrate = [dt, dt_m1](double a, double x_m1)
        {
            // double deltaA = dt + 0.5 * dt_m1;
            double deltaB = 0.5 * (dt + dt_m1);

            auto Val = x_m1 * (1. / dt_m1);
            // auto v   = Val + a * deltaA;
            auto dx = dt * Val + a * deltaB * dt;
            return dx;
        };

        double dx = integrate(a_starx, star.position_m1[0]);
        double dy = integrate(a_stary, star.position_m1[1]);
        double dz = integrate(a_starz, star.position_m1[2]);
        star.position[0] += dx;
        star.position[1] += dy;
        star.position[2] += dz;
        star.position_m1[0] = dx;
        star.position_m1[1] = dy;
        star.position_m1[2] = dz;
    }
    // Send to ranks
    MPI_Bcast(star.position.data(), 3, MpiType<double>{}, 0, MPI_COMM_WORLD);
}

//! @brief For Tuples A, B ... call f(a1, b1 ...), f(a2, b2 ...)
template<typename... Tuples, typename F>
requires(std::tuple_size_v<std::decay_t<Tuples>> == ...) void for_each_tuples(F&& f, Tuples&&... tuples)
{
    auto f_i = [&](auto I) { return f(std::get<I>(std::forward<Tuples>(tuples))...); };

    auto iterate_each = [&f_i]<size_t... Is>(std::index_sequence<Is...>)
    { (f_i(std::integral_constant<size_t, Is>{}), ...); };

    constexpr size_t n_elements = std::min({std::tuple_size_v<std::decay_t<Tuples>>...});
    iterate_each(std::make_index_sequence<n_elements>{});
}

template<typename ActiveFields, typename Dataset, typename Domain, typename StarData>
void accreteParticles(Dataset& d, Domain& domain, StarData& star, double r_outer = 25.)
{
    auto copyContent = [&d](size_t from, size_t to)
    {
        if (from == to) return;
        auto fields_from = cstone::getPointers(get<ActiveFields>(d), from);
        auto fields_to   = cstone::getPointers(get<ActiveFields>(d), to);
        for_each_tuples([]<typename T>(const T* from_ptr, T* to_ptr) { *to_ptr = *from_ptr; }, fields_from, fields_to);
    };

    size_t endIndexNew     = domain.endIndex();
    auto   delete_particle = [&endIndexNew, &domain, copyContent](size_t i)
    {
        assert(endIndexNew > 0);
        assert(endIndexNew - domain.startIndex > 0);
        assert(i > startIndex);

        copyContent(endIndexNew - 1, i);
        endIndexNew--;
    };
    // Go through particles, identify which to delete.
    for (size_t i = domain.startIndex(); i < domain.endIndex(); i++)
    {
        const double x     = d.x[i] - star.position[0];
        const double y     = d.y[i] - star.position[1];
        const double z     = d.z[i] - star.position[2];
        const double dist2 = x * x + y * y + z * z;
        if (dist2 < star.inner_size * star.inner_size || dist2 > r_outer * r_outer)
        {
            star.m += d.m[i];
            double momentum_x = d.vx[i] * d.m[i];
            double momentum_y = d.vy[i] * d.m[i];
            double momentum_z = d.vz[i] * d.m[i];

            star.position_m1[0] += momentum_x / star.m;
            star.position_m1[1] += momentum_y / star.m;
            star.position_m1[2] += momentum_z / star.m;

            delete_particle(i);
        }
    }
    domain.setEndIndex(endIndexNew);
    // If delete, swap with endIndex - 1, decrease endIndex, save newEndIndex.

    // Get momentum and mass of particle and add to star.
}

template<typename Dataset>
void betaCooling(Dataset& d, size_t startIndex, size_t endIndex, const std::array<double, 3>& star_pos,
                 const double stellarMass, const double beta = 3.)
{
    for (size_t i = startIndex; i < endIndex; i++)
    {
        const double x     = d.x[i] - star_pos[0];
        const double y     = d.y[i] - star_pos[1];
        const double z     = d.z[i] - star_pos[2];
        const double dist2 = x * x + y * y + z * z;
        const double dist  = std::sqrt(dist2);
        const double omega = std::sqrt(d.g * stellarMass / (dist2 * dist));
        d.du[i] += -omega / beta;
    }
}
} // namespace planet

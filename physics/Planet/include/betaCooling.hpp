//
// Created by Noah Kubli on 17.04.2024.
//

#pragma once

#include <cmath>
#include "cstone/tree/accel_switch.hpp"
#include "betaCooling_gpu.hpp"
#include "sph/particles_data.hpp"

namespace planet
{

template<typename Tpos, typename Ts>
void betaCoolingImpl(size_t first, size_t last, const Tpos* x, const Tpos* y, const Tpos* z,
                     std::floating_point auto* du, const std::floating_point auto* u, Ts star_mass, const Ts* star_pos,
                     Ts beta, Tpos g, const std::floating_point auto* rho, std::floating_point auto u_floor,
                     std::floating_point auto cooling_rho_limit = 1.683e-3)
{

#pragma omp parallel for
    for (size_t i = first; i < last; i++)
    {
        if (rho[i] < cooling_rho_limit && u[i] > u_floor)
        {
            const double dx    = x[i] - star_pos[0];
            const double dy    = y[i] - star_pos[1];
            const double dz    = z[i] - star_pos[2];
            const double dist2 = dx * dx + dy * dy + dz * dz;
            const double dist  = std::sqrt(dist2);
            const double omega = std::sqrt(g * star_mass / (dist2 * dist));
            du[i] += -u[i] * omega / beta;
        }
    }
}

template<std::floating_point Tt>
Tt duTimestepAndTempFloorImpl(size_t first, size_t last, std::floating_point auto* du, std::floating_point auto* u,
                              const std::floating_point auto* du_m1, std::floating_point auto u_floor,
                              std::floating_point auto u_max, Tt k_u)
{
    size_t n_below_floor{};
    size_t n_above_max{};
    Tt     duTimestepMin = std::numeric_limits<Tt>::infinity();

#pragma omp parallel for reduction(min : duTimestepMin) reduction(+ : n_below_floor) reduction(+ : n_above_max)
    for (size_t i = first; i < last; i++)
    {
        if (u[i] < u_floor)
        {
            u[i]  = u_floor;
            du[i] = std::max(0., du[i]);
            n_below_floor++;
        }
        else if (u[i] > u_max)
        {
            u[i]  = u_max;
            du[i] = std::min(0., du[i]);
            n_above_max++;
        }

        Tt duTimestep = k_u * std::abs(u[i] / du[i]);
        duTimestepMin = std::min(duTimestepMin, duTimestep);
    }
    printf("n_below_floor: %zu\t n_above_max: %zu\n", n_below_floor, n_above_max);
    return duTimestepMin;
}

template<typename Dataset, typename StarData>
void betaCooling(Dataset& d, size_t startIndex, size_t endIndex, const StarData& star)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        betaCoolingGPU(startIndex, endIndex, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z),
                       rawPtr(d.devData.u), rawPtr(d.devData.du), star.m, star.position.data(), star.beta, d.g,
                       rawPtr(d.devData.rho), star.u_floor, star.cooling_rho_limit);
    }
    else
    {
        betaCoolingImpl(startIndex, endIndex, d.x.data(), d.y.data(), d.z.data(), d.du.data(), d.u.data(), star.m,
                        star.position.data(), star.beta, d.g, d.rho.data(), star.u_floor, star.cooling_rho_limit);
    }
}

template<typename Dataset, typename StarData>
void duTimestepAndTempFloor(Dataset& d, size_t startIndex, size_t endIndex, StarData& star)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        transferToHost(d, startIndex, endIndex, {"du", "u"});

        auto dt_u = duTimestepAndTempFloorImpl(startIndex, endIndex, d.du.data(), d.u.data(), d.du_m1.data(),
                                               star.u_floor, star.u_max, star.K_u);

        transferToDevice(d, startIndex, endIndex, {"du", "u"});
        star.t_du = dt_u;
    }
    else
    {
        auto dt_u = duTimestepAndTempFloorImpl(startIndex, endIndex, d.du.data(), d.u.data(), d.du_m1.data(),
                                               star.u_floor, star.u_max, star.K_u);
        star.t_du = dt_u;
    }
}
} // namespace planet

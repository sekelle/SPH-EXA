//
// Created by Noah Kubli on 14.03.2024.
//

#pragma once

#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>
#include <iostream>
#include "cstone/tree/definitions.h"

namespace planet
{

template<typename Dataset, typename StarData>
void computeAccretionConditionImpl(size_t first, size_t last, Dataset& d, StarData& star)
{
    const double star_size2 = star.inner_size * star.inner_size;

    double accr_mass{};
    double accr_mom[3]{};
    size_t n_accreted{};
    size_t n_removed{};

#pragma omp parallel for reduction(+ : accr_mass) reduction(+ : accr_mom[ : 3]) reduction(+ : n_accreted)              \
    reduction(+ : n_removed)
    for (size_t i = first; i < last; i++)
    {
        const double dx    = d.x[i] - star.position[0];
        const double dy    = d.y[i] - star.position[1];
        const double dz    = d.z[i] - star.position[2];
        const double dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 < star_size2)
        {
            // Accrete on star
            d.keys[i] = cstone::removeKey<typename Dataset::KeyType>::value;
            accr_mass += d.m[i];
            accr_mom[0] += d.m[i] * d.vx[i];
            accr_mom[1] += d.m[i] * d.vy[i];
            accr_mom[2] += d.m[i] * d.vz[i];
            n_accreted++;
        }
        else if (d.h[i] > star.removal_limit_h)
        {
            // Remove from system
            d.keys[i] = cstone::removeKey<typename Dataset::KeyType>::value;
            n_removed++;
        }
    }

    star.m_accreted_local    = accr_mass;
    star.p_accreted_local[0] = accr_mom[0];
    star.p_accreted_local[1] = accr_mom[1];
    star.p_accreted_local[2] = accr_mom[2];
    star.n_removed_local     = n_removed;
    star.n_accreted_local    = n_accreted;
}

} // namespace planet

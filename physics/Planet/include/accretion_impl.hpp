//
// Created by Noah Kubli on 14.03.2024.
//

#pragma once

#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>
#include <iostream>

namespace planet
{

template<typename T1, typename Th, typename Tremove, typename T2>
void computeAccretionConditionImpl(size_t first, size_t last, const T1* x, const T1* y, const T1* z, const Th* h,
                                   Tremove* remove, const T2* spos, const T2 star_size)
{
    const double star_size2 = star_size * star_size;

#pragma omp parallel for
    for (size_t i = first; i < last; i++)
    {
        const double dx    = x[i] - spos[0];
        const double dy    = y[i] - spos[1];
        const double dz    = z[i] - spos[2];
        const double dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 < star_size2) { remove[i] = 1; } // Accrete to star
        else if (h[i] > 5.0) { remove[i] = 2; }    // Remove from system
    }
}

template<typename Tremove>
void computeNewOrderImpl(size_t first, size_t last, Tremove* remove, size_t* n_accr, size_t* n_rem2)
{
    std::vector<size_t> index(last - first);
    std::iota(index.begin(), index.end(), first);

    auto       sort_by_removal    = [&remove](size_t i) { return (remove[i] == 0); };
    const auto partition_iterator = std::stable_partition(index.begin(), index.end(), sort_by_removal);
    //*n_removed                    = index.end() - partition_iterator;

    auto       sort_by_removal2 = [&remove](size_t i) { return (remove[i] == 1); };
    const auto rem2_it          = std::stable_partition(partition_iterator, index.end(), sort_by_removal2);

    *n_rem2 = index.end() - rem2_it;
    *n_accr = rem2_it - partition_iterator;

    std::copy(index.begin(), index.end(), remove + first);
}

template<typename T1, typename Torder>
void applyNewOrderImpl(size_t first, size_t last, T1* x, T1* scratch, Torder* order)
{
    for (size_t i = 0; i < last - first; i++)
    {
        scratch[i + first] = x[order[i + first]];
    }

    std::copy(scratch + first, scratch + last, x + first);
}

template<typename Tv, typename Tm, typename Tstar>
void sumMassAndMomentumImpl(size_t first, size_t last, const Tv* vx, const Tv* vy, const Tv* vz, const Tm* m,
                            Tv* scratch, Tstar* m_sum, Tstar* p_sum)
{
    std::transform(vx + first, vx + last, m + first, scratch + first, std::multiplies<Tv>{});
    p_sum[0] = std::accumulate(scratch + first, scratch + last, Tv{0.});

    std::transform(vy + first, vy + last, m + first, scratch + first, std::multiplies<Tv>{});
    p_sum[1] = std::accumulate(scratch + first, scratch + last, Tv{0.});

    std::transform(vz + first, vz + last, m + first, scratch + first, std::multiplies<Tv>{});
    p_sum[2] = std::accumulate(scratch + first, scratch + last, Tv{0.});

    *m_sum = std::accumulate(m + first, m + last, 0.);
}
} // namespace planet

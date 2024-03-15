//
// Created by Noah Kubli on 14.03.2024.
//

#pragma once

#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>

namespace planet
{
template<typename T1, typename Tremove, typename T2>
void computeAccretionConditionImpl(size_t first, size_t last, const T1* x, const T1* y, const T1* z, Tremove* remove,
                                   const T2* spos, const T2 star_size)
{
    const double star_size2 = star_size * star_size;

#pragma omp parallel for
    for (size_t i = first; i < last; i++)
    {
        const double dx    = x[i] - spos[0];
        const double dy    = y[i] - spos[1];
        const double dz    = z[i] - spos[2];
        const double dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 < star_size2) remove[i] = 1;
    }
}

template<typename T1, typename Tremove>
void moveAccretedToEndImpl(size_t first, size_t last, T1* x, Tremove* remove, size_t* n_removed)
{

    auto sort_by_removal      = [r = remove + first, x_first = x + first](T1& x_i) { return (r[&x_i - x_first] == 0); };
    const auto* partition_ptr = std::stable_partition(x + first, x + last, sort_by_removal);
    *n_removed                = (x + last) - partition_ptr;
}

template<typename Tv, typename Tm, typename Tstar>
void sumMassAndMomentumImpl(size_t sum_first, size_t sum_last, const Tv* vx, const Tv* vy, const Tv* vz, const Tm* m,
                            Tstar* m_sum, Tstar* p_sum)
{
    std::vector<Tv> px(sum_last - sum_first);
    std::vector<Tv> py(sum_last - sum_first);
    std::vector<Tv> pz(sum_last - sum_first);

    std::transform(std::execution::par, vx + sum_first, vx + sum_last, m + sum_first, px.begin(),
                   std::multiplies<Tv>{});
    std::transform(vy + sum_first, vy + sum_last, m + sum_first, py.begin(), std::multiplies<Tv>{});
    std::transform(vz + sum_first, vz + sum_last, m + sum_first, pz.begin(), std::multiplies<Tv>{});

    p_sum[0] = std::accumulate(px.begin(), px.end(), 0.);
    p_sum[1] = std::accumulate(py.begin(), py.end(), 0.);
    p_sum[2] = std::accumulate(pz.begin(), pz.end(), 0.);

    *m_sum = std::accumulate(m + sum_first, m + sum_last, 0.);
}
} // namespace planet

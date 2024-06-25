#pragma once

#include "cstone/findneighbors.hpp"

namespace sph
{

using cstone::LocalIndex;

template<class Tc, class T, class KeyType>
void findNeighborsSph(const Tc* x, const Tc* y, const Tc* z, T* h, LocalIndex firstId, LocalIndex lastId,
                      const cstone::Box<Tc>& box, const cstone::OctreeNsView<Tc, KeyType>& treeView, unsigned ng0,
                      unsigned ngmax, unsigned ngmin, LocalIndex* neighbors, unsigned* nc)
{
    LocalIndex numWork = lastId - firstId;

    size_t        numFails     = 0;
    constexpr int maxIteration = 110;

#pragma omp parallel for reduction(+ : numFails)
    for (LocalIndex i = 0; i < numWork; ++i)
    {
        LocalIndex id    = i + firstId;
        unsigned   ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);

        T   h_upper(box.maxExtent());
        T   h_lower{0.};
        int iteration = 0;
        while ((ngmin > ncSph || (ncSph - 1) > ngmax) && iteration++ < maxIteration)
        {
            h_upper = (ncSph - 1) > ngmax ? h[i] : h_upper;
            h_lower = ngmin > ncSph ? h[i] : h_lower;
            if (iteration < 10)
            {
                // Dampen updateH by weighting with proposed smoothing lengths of past iterations
                h[id] = (updateH(ng0, ncSph, h[id]) + h[id] * iteration) / static_cast<T>(iteration + 1);
            }
            else
            {
                // Bisect algorithm
                h[i] = (h_upper + h_lower) / 2.;
            }
            ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);
        }

        numFails += (iteration >= maxIteration);

        nc[i] = ncSph;
    }

    if (numFails)
    {
        std::cout << "Coupled h-neighbor count updated failed to converge for " << numFails << " particles"
                  << std::endl;
    }
}

//! @brief perform neighbor search together with updating the smoothing lengths
template<class T, class Dataset>
void findNeighborsSfc(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { return; }

    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }
    if (d.ngmin > d.ng0) { throw std::runtime_error("ngmin should be smaller than ng0\n"); }


    findNeighborsSph(d.x.data(), d.y.data(), d.z.data(), d.h.data(), startIndex, endIndex, box, d.treeView, d.ng0,
                     d.ngmax, d.ngmin, d.neighbors.data(), d.nc.data() + startIndex);
}

} // namespace sph

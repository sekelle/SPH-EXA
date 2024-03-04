//
// Created by Noah Kubli on 11.03.2024.
//

#pragma once

/*! @file
 * @brief output time and energies each iteration (default)
 * @author Lukas Schmidt
 * @author Noah Kubli
 */

#include <fstream>

#include "conserved_quantities.hpp"
#include "iobservables.hpp"
#include "io/file_utils.hpp"

namespace sphexa
{

template<class Dataset>
class PlanetTimeAndEnergy : public IObservables<Dataset>
{
    std::ostream& constantsFile;
    using T = typename Dataset::RealType;

public:
    explicit PlanetTimeAndEnergy(std::ostream& constPath)
        : constantsFile(constPath)
    {
    }

    void computeAndWrite(Dataset& simData, size_t firstIndex, size_t lastIndex, const cstone::Box<T>& /*box*/) override
    {
        int rank;
        MPI_Comm_rank(simData.comm, &rank);
        auto& d = simData.hydro;

        computeConservedQuantities(firstIndex, lastIndex, d, simData.comm);
        //computeStarPotentialEnergy(firstIndex, lastIndex, d, simData.planet, simData.comm);

        if (rank == 0)
        {
            fileutils::writeColumns(constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav,
                                    d.linmom, d.angmom);
        }
    }
};

} // namespace sphexa

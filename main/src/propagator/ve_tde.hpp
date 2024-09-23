/*! @file
* @brief A Propagator class for a tidal disruption event with gen. volume elements
*
* @author Noah Kubli
*/

#pragma once

#include <variant>

#include "cstone/fields/field_get.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "exchangeStarPosition.hpp"
#include "computeCentralForce.hpp"
#include "star_data.hpp"
#include "accretion.hpp"
#include "betaCooling.hpp"
#include "io/arg_parser.hpp"

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;
using util::FieldList;

template<bool avClean, class DomainType, class DataType>
class TdeVeProp : public Propagator<DomainType, DataType>
{
protected:
   using Base = Propagator<DomainType, DataType>;
   using Base::pmReader;
   using Base::timer;

   using T             = typename DataType::RealType;
   using KeyType       = typename DataType::KeyType;
   using Tmass         = typename DataType::HydroData::Tmass;
   using MultipoleType = ryoanji::CartesianQuadrupole<Tmass>;

   using Acc       = typename DataType::AcceleratorType;
   using MHolder_t = typename cstone::AccelSwitchType<Acc, MultipoleHolderCpu, MultipoleHolderGpu>::template type<
       MultipoleType, DomainType, typename DataType::HydroData>;

   MHolder_t mHolder_;
   GroupData<Acc> groups_;

   StarData  star;
   /*! @brief the list of conserved particles fields with values preserved between iterations
    *
    * x, y, z, h and m are automatically considered conserved and must not be specified in this list
    */
   using ConservedFields = FieldList<"u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "alpha">;

   //! @brief list of dependent fields, these may be used as scratch space during domain sync
   using DependentFields_ = FieldList<"ax", "ay", "az", "prho", "c", "du", "c11", "c12", "c13", "c22", "c23", "c33",
                                      "xm", "kx", "nc", "rho", "p">;

   //! @brief velocity gradient fields will only be allocated when avClean is true
   using GradVFields = FieldList<"dV11", "dV12", "dV13", "dV22", "dV23", "dV33">;

   //! @brief what will be allocated based AV cleaning choice
   using DependentFields =
       std::conditional_t<avClean, decltype(DependentFields_{} + GradVFields{}), decltype(DependentFields_{})>;

public:
    TdeVeProp(std::ostream& output, size_t rank, const InitSettings& settings)
       : Base(output, rank)
   {
       if (avClean && rank == 0) { std::cout << "AV cleaning is activated" << std::endl; }
   }

   [[nodiscard]] std::vector<std::string> conservedFields() const override
   {
       std::vector<std::string> ret{"x", "y", "z", "h", "m"};
       for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFields{}));
       return ret;
   }
   void load(const std::string& initCond, IFileReader* reader) override
   {
       // Read star position from hdf5 File
       std::string path = removeModifiers(initCond);
       if (std::filesystem::exists(path))
       {
           int snapshotIndex = numberAfterSign(initCond, ":");
           reader->setStep(path, snapshotIndex, FileMode::independent);
           star.loadOrStoreAttributes(reader);
           reader->closeStep();
           printf("star position: %lf\t%lf\t%lf\n", star.position[0], star.position[1], star.position[2]);
           printf("star mass: %lf\n", star.m);
       }
   }
   void save(IFileWriter* writer) override { star.loadOrStoreAttributes(writer); }

   void activateFields(DataType& simData) override
   {
       auto& d = simData.hydro;
       //! @brief Fields accessed in domain sync (x,y,z,h,m,keys) are not part of extensible lists.
       d.setConserved("x", "y", "z", "h", "m");
       d.setDependent("keys");
       std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
       std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));

       d.devData.setConserved("x", "y", "z", "h", "m");
       d.devData.setDependent("keys");
       std::apply([&d](auto... f) { d.devData.setConserved(f.value...); }, make_tuple(ConservedFields{}));
       std::apply([&d](auto... f) { d.devData.setDependent(f.value...); }, make_tuple(DependentFields{}));
   }

   void sync(DomainType& domain, DataType& simData) override
   {
       auto& d = simData.hydro;
       if (d.g != 0.0)
       {
           domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                           get<ConservedFields>(d), get<DependentFields>(d));
       }
       else
       {
           domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                       std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d)), get<DependentFields>(d));
       }
       d.treeView = domain.octreeProperties();
   }

   void computeForces(DomainType& domain, DataType& simData) override
   {
       timer.start();
       pmReader.start();
       sync(domain, simData);
       timer.step("domain::sync");

       auto&  d     = simData.hydro;
       d.resizeAcc(domain.nParticlesWithHalos());
       resizeNeighbors(d, domain.nParticles() * d.ngmax);
       size_t first = domain.startIndex();
       size_t last  = domain.endIndex();

       domain.exchangeHalos(std::tie(get<"m">(d)), get<"ax">(d), get<"ay">(d));

       findNeighborsSfc(first, last, d, domain.box());
       computeGroups(first, last, d, domain.box(), groups_);
       timer.step("FindNeighbors");
       pmReader.step();

       computeXMass(groups_.view(), d, domain.box());
       timer.step("XMass");
       domain.exchangeHalos(std::tie(get<"xm">(d)), get<"ax">(d), get<"keys">(d));
       timer.step("mpi::synchronizeHalos");

       release(d, "ay");
       acquire(d, "gradh");
       computeVeDefGradh(groups_.view(), d, domain.box());
       timer.step("Normalization & Gradh");

       computeEOS(first, last, d);
       timer.step("EquationOfState");

       domain.exchangeHalos(get<"vx", "vy", "vz", "prho", "c", "kx">(d), get<"ax">(d), get<"keys">(d));
       timer.step("mpi::synchronizeHalos");

       release(d, "gradh", "az");
       acquire(d, "divv", "curlv");
       computeIadDivvCurlv(groups_.view(), d, domain.box());
       d.minDtRho = rhoTimestep(first, last, d);
       timer.step("IadVelocityDivCurl");

       domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33", "divv">(d), get<"ax">(d), get<"keys">(d));
       timer.step("mpi::synchronizeHalos");

       computeAVswitches(groups_.view(), d, domain.box());
       timer.step("AVswitches");

       if (avClean)
       {
           domain.exchangeHalos(get<"dV11", "dV12", "dV22", "dV23", "dV33", "alpha">(d), get<"ax">(d), get<"keys">(d));
       }
       else { domain.exchangeHalos(std::tie(get<"alpha">(d)), get<"ax">(d), get<"keys">(d)); }
       timer.step("mpi::synchronizeHalos");

       release(d, "divv", "curlv");
       acquire(d, "ay", "az");
       computeMomentumEnergy<avClean>(groups_.view(), nullptr, d, domain.box());
       timer.step("MomentumAndEnergy");
       pmReader.step();

       if (d.g != 0.0)
       {
           auto groups = mHolder_.computeSpatialGroups(d, domain);
           mHolder_.upsweep(d, domain);
           timer.step("Upsweep");
           pmReader.step();
           mHolder_.traverse(groups, d, domain);
           timer.step("Gravity");
           pmReader.step();
       }

       planet::computeCentralForce(simData.hydro, first, last, star);
       timer.step("computeCentralForce");
   }

   void integrate(DomainType& domain, DataType& simData) override
   {
       auto&  d     = simData.hydro;
       size_t first = domain.startIndex();
       size_t last  = domain.endIndex();

       computeTimestep(first, last, d, star.t_du);
       timer.step("Timestep");

       computePositions(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
       updateSmoothingLength(groups_.view(), d);
       timer.step("UpdateQuantities");

       // Accretion uses the keys field to mark particles that have to be removed
       fill(get<"keys">(d), first, last, KeyType{0});

       planet::computeAccretionCondition(first, last, d, star);
//       planet::computeNewOrder(first, last, d, star);
//       planet::applyNewOrder<ConservedFields, DependentFields>(first, last, d);

       planet::exchangeAndAccreteOnStar(star, d.minDt_m1, Base::rank_);

//       domain.setEndIndex(last - star.n_accreted_local - star.n_removed_local);

       timer.step("accreteParticles");

       if (Base::rank_ == 0)
       {
           printf("star position: %lf\t%lf\t%lf\n", star.position[0], star.position[1], star.position[2]);
           printf("star mass: %lf\n", star.m);
           printf("additional pot. erg.: %lf\n", star.potential);
           printf("rank 0: accreted %zu, removed %zu\n", star.n_accreted_local, star.n_removed_local);
       }
   }

   void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                   const cstone::Box<T>& box) override
   {
       auto& d = simData.hydro;
       d.resize(d.accSize());
       auto fieldPointers = d.data();
       auto indicesDone   = d.outputFieldIndices;
       auto namesDone     = d.outputFieldNames;

       auto output = [&]()
       {
           for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
           {
               int fidx = indicesDone[i];
               if (d.isAllocated(fidx))
               {
                   int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                d.outputFieldIndices.begin();
                   transferToHost(d, first, last, {d.fieldNames[fidx]});
                   std::visit([writer, c = column, key = namesDone[i]](auto field)
                              { writer->writeField(key, field->data(), c); },
                              fieldPointers[fidx]);
                   indicesDone.erase(indicesDone.begin() + i);
                   namesDone.erase(namesDone.begin() + i);
               }
           }
       };

       // first output pass: write everything allocated at the end of the step
       output();

       // second output pass: write temporary quantities produced by the EOS
       release(d, "c11", "c12", "c13");
       acquire(d, /*"rho", "p", */"gradh");
       computeEOS(first, last, d);
       output();
       release(d, /*"rho", "p", */"gradh");
       acquire(d, "c11", "c12", "c13");

       // third output pass: recover temporary curlv and divv quantities
       release(d, "prho", "c");
       acquire(d, "divv", "curlv");
       // partial recovery of cij in range [first:last] without halos, which are not needed for divv and curlv
       if (!indicesDone.empty()) { computeIadDivvCurlv(groups_.view(), d, box); }
       output();
       release(d, "divv", "curlv");
       acquire(d, "prho", "c");

       /* The following data is now lost and no longer available in the integration step
        *  c11, c12, c12: halos invalidated
        *  prho, c: destroyed
        */

       if (!indicesDone.empty() && Base::rank_ == 0)
       {
           std::cout << "WARNING: the following fields are not in use and therefore not output: ";
           for (int fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
           {
               std::cout << d.fieldNames[fidx] << ",";
           }
           std::cout << d.fieldNames[indicesDone.back()] << std::endl;
       }
       timer.step("FileOutput");
   }
};

} // namespace sphexa

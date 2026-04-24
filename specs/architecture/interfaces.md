# Architecture: Interfaces and Data Structures

Structural skeleton for the particle species feature. Defines interfaces, data models,
module boundaries, and dependency ordering. NO implementation code — stubs and contracts only.

All elements trace to design.md. Nothing speculative.

## 1. Type Constants

```cpp
// sph/include/sph/types.hpp (extend existing)
namespace sph
{
    //! @brief particle type identifiers
    enum class ParticleType : uint8_t
    {
        darkMatter = 0,  // gravity only
        gas        = 1   // gravity + SPH
    };
}
```

No enum class required — plain uint8_t constants suffice. The enum provides type safety
and documentation. Design.md: "0: dark matter, 1: gas".

## 2. Phase 1: Type field in ParticlesData

Minimal change — add one field to the existing class.

### Changes to `sph/include/sph/particles_data.hpp`

```
Add:     FieldVector<uint8_t> type;          // after 'id' field declaration
Update:  fieldNames — append "type"
Update:  dataTuple() — append 'type' to the tie
```

`FieldVariant` already includes `FieldVector<uint8_t>*` (used by `rung`). No change needed.

Field count goes from 48 to 49. The `static_assert` on tuple/fieldNames size enforces consistency.

### Backward compatibility

All existing initializers must set `type = 1` (gas) for every particle. This is done in each
init function after particle creation, or via a default fill after resize.

## 3. Phase 2: SimulationData Split

### Design constraint

The `get<"fieldname">(d)` accessor pattern relies on `d.dataTuple()` and `d.fieldNames`.
After the split, callers use `get<"x">(d)` for basic fields and `get<"rho">(d.sph)` for
SPH fields — OR — `SimulationData` provides a unified `dataTuple()` / `fieldNames` that
spans both basic and SPH fields.

**Decision**: SimulationData provides unified access. It inherits from `FieldStates` and
exposes a combined `dataTuple()` and `fieldNames` that covers all fields (basic + SPH).
Internally, fields are organized into basic and SPH members, but externally the interface
is unchanged. This minimizes propagator changes — `get<"x">(d)` and `get<"rho">(d)` both
work on the same `d`.

### New class hierarchy

```
SimulationData<AccType>
├── Basic fields:   keys, x, y, z, x_m1..z_m1, m, h, vx..vz, ax..az, rung, id, type
├── SPH fields:     rho, temp, u, du, du_m1, p, prho, tdpdTrho, c, cv, mue, mui,
│                   divv, curlv, c11..c33, alpha, xm, kx, gradh, dV11..dV33, nc, ugrav, wh, whd
├── General scalars (from basic): iteration, numParticlesGlobal, ttot, minDt, g, eps, ng0, ngmax, ...
├── SPH scalars:    gamma, Kcour, Krho, minDtCourant, minDtRho, alphamin..alphamax, sincIndex, K, ...
├── chem:           ChemistryData<RealType> (unchanged)
├── comm:           MPI_Comm (unchanged)
├── Data structures: neighbors, treeView, traversalStack
├── dataTuple()  →  combined tuple of all FieldVectors (basic + SPH), same order as fieldNames
├── fieldNames   →  combined array of all field name strings
└── FieldStates  →  activation/deactivation for all fields
```

### Propagator template parameter change

```
Before: Propagator<DomainType, SimulationData<AccType>>
         - computeForces operates on simData, accesses simData.hydro
After:  Propagator<DomainType, SimulationData<AccType>>
         - computeForces operates on simData directly
         - simData provides get<"field">() access to all fields
```

The propagator interface (`ipropagator.hpp`) signature does NOT change. The `ParticleDataType`
template parameter is still `SimulationData<AccType>`. What changes is that propagators access
fields on `simData` directly instead of `simData.hydro`.

### loadOrStoreAttributes split

Attributes split between basic (always) and SPH (when SPH is active):
- Basic attributes: iteration, numParticlesGlobal, time, minDt, gravConstant, ng0, ngmax, eps, etaAcc
- SPH attributes: gamma, Kcour, Krho, eosChoice, muiConst, sincIndex, kernelChoice, alphamin, ...

## 4. Phase 3: Type Partition and Gas Layout

### New header: `domain/include/cstone/domain/type_partition.hpp`

```cpp
namespace cstone
{
    //! @brief Stable partition of particles by type, preserving SFC order within each group.
    //!        Returns the permutation vector for reverse operation.
    //! @param type      particle type array (size N)
    //! @param layout    octree layout (unused for global partition, but defines particle range)
    //! @param first     first particle index to partition
    //! @param last      one past last particle index
    //! @param scratch   buffer of size >= (last - first) for temporary storage
    //! @return          permutation vector mapping Order-2 indices → Order-1 indices
    template<class IntVec, class IndexVec>
    IndexVec stablePartitionByType(const IntVec& type,
                                   LocalIndex first, LocalIndex last,
                                   IndexVec& scratch);

    //! @brief Apply a permutation to all arrays in a tuple
    template<class IndexVec, class... Arrays>
    void applyPermutation(const IndexVec& permutation,
                          LocalIndex first, LocalIndex last,
                          Arrays&... arrays);

    //! @brief Reverse a permutation (apply inverse)
    template<class IndexVec, class... Arrays>
    void reversePermutation(const IndexVec& permutation,
                            LocalIndex first, LocalIndex last,
                            Arrays&... arrays);

    //! @brief Find assigned range within a type-partitioned sub-array by SFC key comparison
    //! @param keys          SFC keys of the sub-array (already sorted)
    //! @param subStart      first index of the sub-array in the full array
    //! @param subEnd        one past last index
    //! @param domainKeyStart domain SFC key range start
    //! @param domainKeyEnd   domain SFC key range end
    //! @return              {assignedStart, assignedEnd} within the sub-array
    template<class KeyType>
    std::pair<LocalIndex, LocalIndex>
    findAssignedRange(const KeyType* keys,
                      LocalIndex subStart, LocalIndex subEnd,
                      KeyType domainKeyStart, KeyType domainKeyEnd);

    //! @brief Build a layout for a contiguous sub-array of particles
    //!        Counts particles per leaf cell by scanning SFC keys against cell boundaries
    //! @param keys       SFC keys of the sub-array (sorted)
    //! @param subStart   first index in full array
    //! @param subEnd     one past last index
    //! @param leaves     octree leaf SFC key boundaries (size numLeaves + 1)
    //! @param numLeaves  number of leaf cells
    //! @param layout     output layout array (size numLeaves + 1)
    template<class KeyType>
    void buildSubLayout(const KeyType* keys,
                        LocalIndex subStart, LocalIndex subEnd,
                        const KeyType* leaves, TreeNodeIndex numLeaves,
                        LocalIndex* layout);

    //! @brief Build halo exchange lists for a sub-array from existing halo information
    //! @param subLayout       layout for the sub-array
    //! @param fullLayout      layout for the full array (from domain)
    //! @param incomingHalos   existing incoming halo index list (Order-1)
    //! @param outgoingHalos   existing outgoing halo index list (Order-1)
    //! @param permutation     Order-1 → Order-2 permutation
    //! @return                {subIncoming, subOutgoing} halo lists for the sub-array
    // Implementation details TBD during Phase 3 implementation
}
```

### Gas-only OctreeNsView

No new struct — just copy existing `OctreeNsView` and swap the `layout` pointer:

```cpp
// In propagator code:
auto gasTreeView = d.treeView;           // copy struct (all pointers)
gasTreeView.layout = gasLayout.data();   // swap layout pointer to gas-only layout
```

## 5. Phase 4: HydroDarkProp

### New header: `main/src/propagator/hydro_dark.hpp`

```cpp
namespace sphexa
{
    //! @brief Propagator for mixed dark matter + gas simulations.
    //!        Gravity for all particles (Order-1), SPH for gas only (Order-2).
    template<bool avClean, class DomainType, class DataType>
    class HydroDarkProp : public Propagator<DomainType, DataType>
    {
        using Base = Propagator<DomainType, DataType>;
        using Acc  = typename DataType::AcceleratorType;
        using T    = typename DataType::RealType;
        using KeyType = typename DataType::KeyType;

        // Gravity holder (same as HydroVeProp)
        MultipoleHolder<...> mHolder_;

        // Partition state (persisted across substeps for BDT)
        AccVector<cstone::LocalIndex> permutation_;    // Order-1 → Order-2 mapping

        // Gas sub-array state
        cstone::LocalIndex gasStart_, gasAssignedStart_, gasAssignedEnd_, gasEnd_;
        AccVector<cstone::LocalIndex> gasLayout_;       // per-cell gas particle offsets
        cstone::OctreeNsView<T, KeyType> gasTreeView_;  // tree with gas layout

        // Gas-only halo exchange state
        // SendList/RecvList for gas particles (derived from gasLayout_ + halo peers)

    public:
        std::vector<std::string> conservedFields() const override;
        void activateFields(DataType& d) override;
        void sync(DomainType& domain, DataType& d) override;
        void computeForces(DomainType& domain, DataType& d) override;
        void integrate(DomainType& domain, DataType& d) override;

    private:
        //! @brief Step 3: pass 1 neighbor search (Order-1, all types, sets h)
        void neighborSearchPass1(DataType& d, const cstone::Box<T>& box);

        //! @brief Steps 6-10: partition → gas layout → gas halo lists → pass 2 neighbors
        void partitionAndBuildGasState(DataType& d, const DomainType& domain);

        //! @brief Step 11: gas-only neighbor search (Order-2, gas only)
        void neighborSearchPass2(DataType& d, const cstone::Box<T>& box);

        //! @brief Step 12: SPH kernel sub-steps with gas-only halo exchanges
        void computeSphForces(DataType& d, DomainType& domain);

        //! @brief Step 13: reverse partition back to Order-1
        void reversePartition(DataType& d);

        //! @brief Gas-only halo exchange (replaces domain.exchangeHalos for gas fields)
        template<class... Vectors>
        void exchangeGasHalos(std::tuple<Vectors&...> arrays, ...);
    };
}
```

### Factory registration

```cpp
// main/src/propagator/factory.hpp — add:
if (choice == "hydro-dark") { return PropLib<DomainType, ParticleDataType>::makeHydroDarkProp(output, rank, avClean); }
```

### Acceleration accumulation change

```
// sph/include/sph/hydro_ve/momentum_energy_kern.hpp — change:
grad_P_x[i] = -K * momentum_x;   →   grad_P_x[i] += -K * momentum_x;
grad_P_y[i] = -K * momentum_y;   →   grad_P_y[i] += -K * momentum_y;
grad_P_z[i] = -K * momentum_z;   →   grad_P_z[i] += -K * momentum_z;
// du[i] = ... stays as assignment (no change)

// Same change in GPU kernel: sph/include/sph/hydro_ve/momentum_energy_gpu.cu
// Same change in std variant: sph/include/sph/hydro_std/momentum_energy_kern.hpp + GPU
```

**Note**: This change affects ALL propagators. Existing propagators must zero ax/ay/az before
calling momentum+energy. Currently SPH writes first (=), then gravity adds (+=). After the
change, both use +=, so zeroing must happen before either. Verify all existing propagators
include a zero step (NbodyProp already does; others need it added before the SPH phase).

## 6. Module Dependency Graph

```
Phase 1: sph/particles_data.hpp ← all propagators, all inits, all I/O
Phase 2: sph/particles_data.hpp + main/simulation_data.hpp ← all propagators
Phase 3: domain/type_partition.hpp ← main/propagator/hydro_dark.hpp
Phase 4: main/propagator/hydro_dark.hpp ← main/propagator/factory.hpp
Phase 5: main/init/*_init.hpp ← main/sphexa.cpp
```

No circular dependencies. Each phase extends without breaking prior phases.

## 7. Build Phase Ordering

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
```

Each phase is independently testable. Phase 1 can be merged alone. Phase 2 requires Phase 1.
Phase 3 is independent of Phase 2 (operates on raw arrays, not the class hierarchy), but
Phase 4 requires both Phase 2 and Phase 3. Phase 5 requires Phase 1 (type field in I/O).

Recommended parallel work:
- Phase 1 first (small, enables everything)
- Phase 2 and Phase 3 can proceed in parallel after Phase 1
- Phase 4 after both Phase 2 and Phase 3
- Phase 5 after Phase 1 (can proceed in parallel with 2-4)

## 8. Files Changed Per Phase

### Phase 1 (type field)
- `sph/include/sph/particles_data.hpp` — add type field, update fieldNames, dataTuple
- `main/src/init/*_init.hpp` — set type=1 in each initializer
- `main/src/io/ifile_io_hdf5.cpp` — handle type in read/write (if not automatic)

### Phase 2 (SimulationData split)
- `sph/include/sph/particles_data.hpp` — remove basic fields + general scalars
- `main/src/sphexa/simulation_data.hpp` — absorb basic fields, unified dataTuple/fieldNames
- `main/src/propagator/*.hpp` — change `simData.hydro` → `simData` for basic field access
- `main/src/propagator/ipropagator.hpp` — update printIterationTimings references
- `main/src/sphexa/sphexa.cpp` — update main loop references

### Phase 3 (type partition)
- `domain/include/cstone/domain/type_partition.hpp` — NEW: partition functions
- `domain/test/unit/domain/type_partition.cpp` — NEW: unit tests
- `domain/test/unit_cuda/domain/type_partition_gpu.cu` — NEW: GPU tests

### Phase 4 (HydroDarkProp)
- `main/src/propagator/hydro_dark.hpp` — NEW: propagator
- `main/src/propagator/factory.hpp` — register "hydro-dark"
- `main/src/propagator/propagator.h` — add makeHydroDarkProp
- `sph/include/sph/hydro_ve/momentum_energy_kern.hpp` — ax = → ax +=
- `sph/include/sph/hydro_ve/momentum_energy_gpu.cu` — ax = → ax += (GPU)
- `sph/include/sph/hydro_std/momentum_energy_kern.hpp` — ax = → ax +=
- `sph/include/sph/hydro_std/momentum_energy_gpu.cu` — ax = → ax += (GPU)
- All existing propagators — add zero ax/ay/az before SPH phase

### Phase 5 (mixed init)
- `main/src/init/file_init.hpp` — read type column from HDF5/ASCII
- `main/src/io/ifile_io_hdf5.cpp` — type field read with fallback to gas default

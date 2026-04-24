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

## 3. Phase 2: Two Container Classes

### Design principle

**No unified dataTuple/fieldNames on SimulationData.** Each container class is self-contained
with its own `dataTuple()`, `fieldNames`, and `FieldStates` inheritance.

**No `get<"fieldname">()` accessor pattern.** All field access is direct member access:
`simData.hydro.x`, `simData.hydro.rho`, etc. The `get<>` pattern is eliminated from all
propagator code and replaced with explicit member access.

### Two container classes

**DarkData** (basic/gravity fields — all particle types):

```cpp
// sph/include/sph/dark_data.hpp
template<class AccType>
class DarkData : public cstone::FieldStates<DarkData<AccType>>
{
public:
    using AcceleratorType = AccType;
    using KeyType   = sph::SphTypes::KeyType;
    using RealType  = sph::SphTypes::CoordinateType;
    using HydroType = sph::SphTypes::HydroType;
    using XM1Type   = sph::SphTypes::XM1Type;
    using Tmass     = sph::SphTypes::Tmass;

    template<class ValueType>
    using FieldVector = std::conditional_t<cstone::HaveGpu<AccType>{},
                                           cstone::DeviceVector<ValueType>,
                                           std::vector<ValueType>>;

    using FieldVariant = std::variant<FieldVector<float>*, FieldVector<double>*,
                                      FieldVector<unsigned>*, FieldVector<uint64_t>*,
                                      FieldVector<uint8_t>*>;

    // --- FieldVectors ---
    FieldVector<RealType>  x, y, z;
    FieldVector<XM1Type>   x_m1, y_m1, z_m1;
    FieldVector<HydroType> vx, vy, vz;
    FieldVector<HydroType> h;
    FieldVector<Tmass>     m;
    FieldVector<HydroType> ax, ay, az;
    FieldVector<KeyType>   keys;
    FieldVector<uint8_t>   rung;
    FieldVector<uint64_t>  id;
    FieldVector<uint8_t>   type;

    // --- General simulation scalars (per-container) ---
    uint64_t iteration{1};
    RealType ttot{0.0}, etot{0.0}, ecin{0.0}, eint{0.0}, egrav{0.0};
    RealType linmom{0.0}, angmom{0.0};
    RealType minDt{1e-12}, minDt_m1{1e-12};
    RealType g{0.0}, eps{0.005}, etaAcc{0.2};
    unsigned ng0{100}, ngmax{150};
    int removeUnconvergedParticles{false};
    constexpr static RealType maxDtIncrease = 1.1;
    uint64_t localNeighbors{0}, maxHalos{0};
    size_t stackUsedNc{0}, stackUsedGravity{0};

    // --- Data structures ---
    std::vector<cstone::LocalIndex> neighbors;
    cstone::OctreeNsView<RealType, KeyType> treeView;
    FieldVector<cstone::LocalIndex> traversalStack;

    // --- I/O ---
    std::vector<int> outputFieldIndices;
    std::vector<std::string> outputFieldNames;
    static const inline std::string prefix{};

    // --- Field registry (for FieldStates, I/O, domain sync) ---
    inline static constexpr std::array fieldNames{
        "x", "y", "z", "x_m1", "y_m1", "z_m1", "vx", "vy", "vz",
        "h", "m", "ax", "ay", "az", "keys", "rung", "id", "type"};

    auto dataTuple() { return std::tie(x, y, z, x_m1, y_m1, z_m1, vx, vy, vz,
                                        h, m, ax, ay, az, keys, rung, id, type); }
    auto data() { /* FieldVariant array from dataTuple */ }
    void resize(size_t size) { /* same pattern as current ParticlesData */ }
    size_t size() { /* same pattern */ }
    void setOutputFields(std::vector<std::string>& outFields) { /* same pattern */ }
    void loadOrStoreAttributes(Archive* ar) { /* basic attributes only */ }

private:
    float allocGrowthRate_{1.05};
};
```

**SphData** (SPH-specific fields — gas particles only):

```cpp
// sph/include/sph/sph_data.hpp
template<class AccType>
class SphData : public cstone::FieldStates<SphData<AccType>>
{
public:
    using AcceleratorType = AccType;
    // ... same type aliases, FieldVector, FieldVariant ...

    // --- FieldVectors ---
    FieldVector<HydroType> rho;
    FieldVector<RealType>  temp, u;
    FieldVector<HydroType> p, prho, tdpdTrho;
    FieldVector<HydroType> c, cv;
    FieldVector<HydroType> mue, mui;
    FieldVector<HydroType> divv, curlv;
    FieldVector<HydroType> ugrav;
    FieldVector<RealType>  du;
    FieldVector<XM1Type>   du_m1;
    FieldVector<HydroType> c11, c12, c13, c22, c23, c33;
    FieldVector<HydroType> alpha;
    FieldVector<HydroType> xm, kx, gradh;
    FieldVector<unsigned>  nc;
    FieldVector<HydroType> dV11, dV12, dV13, dV22, dV23, dV33;

    // --- SPH kernel lookup tables ---
    FieldVector<HydroType> wh, whd;

    // --- SPH-specific scalars ---
    sph::EosType eosChoice{sph::EosType::idealGas};
    RealType gamma{5.0 / 3.0};
    RealType polytropic_index{5. / 3.}, polytropic_const{1.};
    Tmass muiConst{10.0};
    HydroType soundSpeedConst{1.0};
    RealType Kcour{0.2}, Krho{0.06};
    RealType minDtCourant{INFINITY}, minDtRho{INFINITY};
    HydroType alphamin{0.05}, alphamax{1.0}, decay_constant{0.2};
    constexpr static HydroType Atmin = 0.1, Atmax = 0.2;
    constexpr static HydroType ramp = 1.0 / (Atmax - Atmin);
    RealType sincIndex{6.0};
    sph::SphKernelType kernelChoice{sph::SphKernelType::sinc_n};
    RealType K{0};

    // --- Field registry ---
    inline static constexpr std::array fieldNames{
        "rho", "temp", "u", "p", "prho", "tdpdTrho", "c", "cv", "mue", "mui",
        "divv", "curlv", "ugrav", "du", "du_m1", "c11", "c12", "c13", "c22", "c23", "c33",
        "alpha", "xm", "kx", "gradh", "nc", "dV11", "dV12", "dV13", "dV22", "dV23", "dV33"};

    auto dataTuple() { return std::tie(rho, temp, u, p, prho, tdpdTrho, c, cv, mue, mui,
                                        divv, curlv, ugrav, du, du_m1,
                                        c11, c12, c13, c22, c23, c33,
                                        alpha, xm, kx, gradh, nc,
                                        dV11, dV12, dV13, dV22, dV23, dV33); }
    auto data() { /* FieldVariant array from dataTuple */ }
    void resize(size_t size) { /* same pattern */ }
    void setOutputFields(std::vector<std::string>& outFields) { /* same pattern */ }
    void loadOrStoreAttributes(Archive* ar) { /* SPH attributes only */ }
    void createTables() { /* compute K, wh, whd from sincIndex/kernelChoice */ }

private:
    float allocGrowthRate_{1.05};
};
```

### SimulationData composition

```cpp
// main/src/sphexa/simulation_data.hpp
template<class AccType>
class SimulationData
{
public:
    using AcceleratorType = AccType;
    using KeyType  = sph::SphTypes::KeyType;
    using RealType = sph::SphTypes::CoordinateType;

    //! @brief basic fields: positions, velocities, mass, keys, type — all particle types
    DarkData<AccType> dark;

    //! @brief SPH fields: density, pressure, energy, etc. — gas particles only
    SphData<AccType> sph;

    //! @brief chemistry data for radiative cooling
    cooling::ChemistryData<RealType> chem;

    MPI_Comm comm;

    //! @brief global counters (shared across containers, used by printIterationTimings)
    uint64_t numParticlesGlobal{0}, numParticlesGlobalPrev{};
    uint64_t totalNeighbors{0};

    //! @brief resize both containers together (INV-13: uniform array sizes)
    void resize(size_t n) { dark.resize(n); sph.resize(n); }

    void setOutputFields(std::vector<std::string> outFields)
    {
        dark.setOutputFields(outFields);
        sph.setOutputFields(outFields);
        chem.setOutputFields(outFields);
        // report unrecognized fields
    }
};
```

### SPH kernel view object

SPH kernels currently take a `Dataset& d` template parameter and access fields from both
what will become `DarkData` (x, y, z, h, m, neighbors, treeView) and `SphData` (rho, xm,
c, nc, etc.) through `d.x`, `d.rho`, etc.

After the split, a non-owning **view struct** provides the combined interface:

```cpp
// sph/include/sph/particle_view.hpp
template<class AccType>
struct ParticleView
{
    using RealType  = sph::SphTypes::CoordinateType;
    using HydroType = sph::SphTypes::HydroType;
    using KeyType   = sph::SphTypes::KeyType;
    using Tmass     = sph::SphTypes::Tmass;
    using XM1Type   = sph::SphTypes::XM1Type;

    template<class ValueType>
    using FieldVector = typename DarkData<AccType>::template FieldVector<ValueType>;

    // --- Non-owning pointers/references from DarkData ---
    FieldVector<RealType>&  x; FieldVector<RealType>&  y; FieldVector<RealType>&  z;
    FieldVector<HydroType>& h;
    FieldVector<Tmass>&     m;
    FieldVector<HydroType>& vx; FieldVector<HydroType>& vy; FieldVector<HydroType>& vz;
    FieldVector<HydroType>& ax; FieldVector<HydroType>& ay; FieldVector<HydroType>& az;
    FieldVector<KeyType>&   keys;
    unsigned ng0, ngmax;
    std::vector<cstone::LocalIndex>& neighbors;
    cstone::OctreeNsView<RealType, KeyType>& treeView;

    // --- Non-owning pointers/references from SphData ---
    FieldVector<HydroType>& rho;
    FieldVector<RealType>&  u;
    FieldVector<HydroType>& p; FieldVector<HydroType>& prho; FieldVector<HydroType>& tdpdTrho;
    FieldVector<HydroType>& c; FieldVector<HydroType>& cv;
    FieldVector<HydroType>& mue; FieldVector<HydroType>& mui;
    FieldVector<RealType>&  temp;
    FieldVector<HydroType>& divv; FieldVector<HydroType>& curlv;
    FieldVector<RealType>&  du;
    FieldVector<XM1Type>&   du_m1;
    FieldVector<HydroType>& c11; FieldVector<HydroType>& c12; FieldVector<HydroType>& c13;
    FieldVector<HydroType>& c22; FieldVector<HydroType>& c23; FieldVector<HydroType>& c33;
    FieldVector<HydroType>& alpha;
    FieldVector<HydroType>& xm; FieldVector<HydroType>& kx; FieldVector<HydroType>& gradh;
    FieldVector<unsigned>&  nc;
    FieldVector<HydroType>& dV11; FieldVector<HydroType>& dV12; FieldVector<HydroType>& dV13;
    FieldVector<HydroType>& dV22; FieldVector<HydroType>& dV23; FieldVector<HydroType>& dV33;
    FieldVector<HydroType>& wh; FieldVector<HydroType>& whd;

    // --- SPH scalars ---
    RealType K;
    HydroType Atmin, Atmax, ramp;
    HydroType alphamin, alphamax, decay_constant;
    // ... other SPH scalars as needed by kernels
};
```

**Construction** from SimulationData (in propagator code):
```cpp
auto view = ParticleView<AccType>{
    simData.dark.x, simData.dark.y, simData.dark.z,
    simData.dark.h, simData.dark.m,
    simData.dark.vx, simData.dark.vy, simData.dark.vz,
    simData.dark.ax, simData.dark.ay, simData.dark.az,
    simData.dark.keys, simData.dark.ng0, simData.dark.ngmax,
    simData.dark.neighbors, simData.dark.treeView,
    simData.sph.rho, simData.sph.u, simData.sph.p, ...
    simData.sph.K, simData.sph.Atmin, ...
};
```

Or provide a factory function:
```cpp
auto view = makeParticleView(simData);  // constructs from dark + sph
```

SPH kernel signatures stay unchanged — they still take `Dataset& d` and access `d.x`,
`d.rho`, etc. The `Dataset` is now `ParticleView<AccType>` instead of `ParticlesData<AccType>`.
No kernel code changes needed beyond the template instantiation.

### Field access in propagators

Propagator code accesses containers directly (no `get<>`):
```cpp
// Before (get<> accessor on simData.hydro):
auto& d = simData.hydro;
get<"x">(d)        →  simData.dark.x
get<"rho">(d)      →  simData.sph.rho
get<"ax">(d)       →  simData.dark.ax
d.minDt            →  simData.dark.minDt
d.gamma            →  simData.sph.gamma

// Domain sync:
domain.syncGrav(get<"keys">(d), get<"x">(d), ...)
→  domain.syncGrav(simData.dark.keys, simData.dark.x, ...)

// Halo exchange:
domain.exchangeHalos(get<"xm">(d), ...)
→  domain.exchangeHalos(std::tie(simData.sph.xm), ...)

// SPH kernel calls via view:
auto view = makeParticleView(simData);
computeXMass(first, last, view, box);
```

### printIterationTimings

`printIterationTimings` no longer takes the full `SimulationData`. Instead it receives
specific values:

```cpp
void printIterationTimings(const DomainType& domain,
                           uint64_t iteration, uint64_t numParticlesGlobal,
                           uint64_t numParticlesGlobalPrev,
                           uint64_t totalNeighbors, uint64_t maxHalos,
                           RealType ttot, RealType minDt,
                           RealType etot, RealType eint, RealType ecin, RealType egrav,
                           size_t stackUsedNc, size_t stackUsedGravity);
```

Callers construct the argument list from `simData.*`, `simData.dark.*`, and `simData.sph.*`
as needed. The function has no dependency on container class layout.

### Domain sync with two containers

`domain.sync()` and `domain.syncGrav()` accept particle properties as tuples.
Both dark and sph fields must be passed:

```cpp
domain.syncGrav(simData.dark.keys,
                simData.dark.x, simData.dark.y, simData.dark.z,
                simData.dark.h, simData.dark.m,
                std::tuple_cat(darkConservedTuple, sphConservedTuple),
                std::tuple_cat(darkDependentTuple, sphDependentTuple));
```

The conserved/dependent tuples are built by each propagator from the fields it activates.
Both containers' fields are included to ensure uniform array sizes (INV-13).

### Resize coordination

Use `SimulationData::resize()` to keep containers in sync:
```cpp
simData.resize(newSize);  // resizes dark + sph together (INV-13)
```

### loadOrStoreAttributes split

Each container handles its own attributes:
```cpp
simData.dark.loadOrStoreAttributes(ar);  // iteration, time, minDt, g, ng0, ...
simData.sph.loadOrStoreAttributes(ar);   // gamma, Kcour, sincIndex, ...
```

## 4. Phase 3: Type Partition and Gas Layout

### New header: `domain/include/cstone/domain/type_partition.hpp`

```cpp
namespace cstone
{
    //! @brief Stable partition of particles by type, preserving SFC order within each group.
    //!        Returns the permutation vector for reverse operation.
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
    template<class KeyType>
    std::pair<LocalIndex, LocalIndex>
    findAssignedRange(const KeyType* keys,
                      LocalIndex subStart, LocalIndex subEnd,
                      KeyType domainKeyStart, KeyType domainKeyEnd);

    //! @brief Build a layout for a contiguous sub-array of particles
    template<class KeyType>
    void buildSubLayout(const KeyType* keys,
                        LocalIndex subStart, LocalIndex subEnd,
                        const KeyType* leaves, TreeNodeIndex numLeaves,
                        LocalIndex* layout);
}
```

### Partition operates on both containers

The partition permutation must be applied to ALL field arrays from both `DarkData` and `SphData`:

```cpp
auto perm = stablePartitionByType(simData.dark.type, 0, totalSize, scratch);

// Apply to all dark fields
applyPermutation(perm, 0, totalSize, simData.dark.x, simData.dark.y, simData.dark.z,
                 simData.dark.h, simData.dark.m, simData.dark.vx, ...);

// Apply to all active sph fields
applyPermutation(perm, 0, totalSize, simData.sph.rho, simData.sph.u, simData.sph.p, ...);
```

Alternatively, use `dataTuple()` from each container to apply permutation to all fields at once.

### Gas-only OctreeNsView

Copy existing treeView, swap layout pointer:
```cpp
auto gasTreeView = simData.dark.treeView;
gasTreeView.layout = gasLayout.data();
```

## 5. Phase 4: HydroDarkProp

### New header: `main/src/propagator/hydro_dark.hpp`

```cpp
namespace sphexa
{
    template<bool avClean, class DomainType, class DataType>
    class HydroDarkProp : public Propagator<DomainType, DataType>
    {
        using Base    = Propagator<DomainType, DataType>;
        using Acc     = typename DataType::AcceleratorType;
        using T       = typename DataType::RealType;
        using KeyType = typename DataType::KeyType;

        // Gravity holder
        MultipoleHolder<...> mHolder_;

        // Partition state
        AccVector<cstone::LocalIndex> permutation_;

        // Gas sub-array state
        cstone::LocalIndex gasStart_, gasAssignedStart_, gasAssignedEnd_, gasEnd_;
        AccVector<cstone::LocalIndex> gasLayout_;
        cstone::OctreeNsView<T, KeyType> gasTreeView_;

        // Gas-only halo exchange state
        // SendList/RecvList for gas particles

    public:
        std::vector<std::string> conservedFields() const override;
        void activateFields(DataType& d) override;
        void sync(DomainType& domain, DataType& d) override;
        void computeForces(DomainType& domain, DataType& d) override;
        void integrate(DomainType& domain, DataType& d) override;
    };
}
```

### SPH kernel calls via ParticleView

SPH kernels are called with a `ParticleView` constructed from both containers:

```cpp
auto view = makeParticleView(simData);
computeXMass(first, last, view, box);       // view.x, view.m, view.xm all work
computeVe(first, last, view, box);          // view.kx, view.xm, etc.
computeMomentumEnergy<avClean>(first, last, view, box);  // view.ax += ..., view.du = ...
```

Kernel template parameter `Dataset` is instantiated as `ParticleView<AccType>` instead of
`ParticlesData<AccType>`. No kernel source code changes needed — they still access `d.x`,
`d.rho`, etc. via the same member names.

### Acceleration accumulation change

```
// sph/include/sph/hydro_ve/momentum_energy_kern.hpp:
grad_P_x[i] = -K * momentum_x;   →   grad_P_x[i] += -K * momentum_x;
grad_P_y[i] = -K * momentum_y;   →   grad_P_y[i] += -K * momentum_y;
grad_P_z[i] = -K * momentum_z;   →   grad_P_z[i] += -K * momentum_z;
// du[i] = ... stays as assignment

// Same in GPU kernels and std variants
```

All existing propagators must zero `ax, ay, az` before calling SPH momentum kernels.

### Factory registration

```cpp
// main/src/propagator/factory.hpp:
if (choice == "hydro-dark") {
    return PropLib<DomainType, ParticleDataType>::makeHydroDarkProp(output, rank, avClean);
}
```

## 6. Module Dependency Graph

```
Phase 1: sph/particles_data.hpp ← all propagators, all inits, all I/O
Phase 2: sph/dark_data.hpp (NEW) + sph/sph_data.hpp (NEW) + main/simulation_data.hpp
         ← all propagators (field access rewrite)
Phase 3: domain/type_partition.hpp (NEW) ← main/propagator/hydro_dark.hpp
Phase 4: main/propagator/hydro_dark.hpp (NEW) ← main/propagator/factory.hpp
Phase 5: main/init/*_init.hpp ← main/sphexa.cpp
```

No circular dependencies. Each phase extends without breaking prior phases.

## 7. Build Phase Ordering

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
          ↘ (Phase 3 can run parallel with Phase 2)     ↗
            Phase 5 can start after Phase 1
```

Phase 3 (type_partition.hpp) has no dependency on the container split (Phase 2).
It operates on raw arrays and could be developed in parallel with Phase 2.
Phase 4 requires both Phase 2 and Phase 3.
Phase 5 requires Phase 1 only.

## 8. Files Changed Per Phase

### Phase 1 (type field)
- `sph/include/sph/particles_data.hpp` — add type field, update fieldNames, dataTuple
- `main/src/init/*_init.hpp` — set type=1 in each initializer
- `main/src/io/ifile_io_hdf5.cpp` — handle type in read/write (if not automatic)

### Phase 2 (container split)
- `sph/include/sph/dark_data.hpp` — NEW: basic/gravity fields container
- `sph/include/sph/sph_data.hpp` — NEW: SPH fields container
- `sph/include/sph/particle_view.hpp` — NEW: non-owning view combining dark + sph fields
- `sph/include/sph/particles_data.hpp` — REMOVE or reduce to backward-compat alias
- `main/src/sphexa/simulation_data.hpp` — dark + sph composition, resize(), global counters
- `main/src/propagator/*.hpp` — replace `get<>` with direct `simData.dark.*` / `simData.sph.*`,
  construct ParticleView for SPH kernel calls
- `main/src/propagator/ipropagator.hpp` — printIterationTimings takes explicit values,
  outputAllocatedFields iterates dark.data() + sph.data() + chem.data()
- `main/src/sphexa/sphexa.cpp` — update main loop references
- `domain/include/cstone/fields/field_get.hpp` — no longer needed by propagators

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

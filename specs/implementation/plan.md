# Implementation Plan: Particle Species

Step-by-step implementation plan with correctness verification for each phase.
Each step ends with a concrete test that proves the step is correct before proceeding.

## General approach

- TDD: write failing test, implement, pass, verify no regressions
- Each step is a single commit with passing tests
- Run full existing test suite after every step to catch regressions
- Steps within a phase are ordered by dependency

---

## Phase 1: Add the type field

**Goal**: Add `type` as a `FieldVector<uint8_t>` to `ParticlesData`. All existing behavior unchanged.

### Step 1.1: Add type field to ParticlesData

**Do**:
- Add `FieldVector<uint8_t> type;` after `id` in `particles_data.hpp`
- Append `"type"` to `fieldNames`
- Append `type` to `dataTuple()`
- The `static_assert` on tuple/fieldNames size enforces consistency at compile time

**Test** (write first):
- `TEST(TypeField, existsInFieldRegistry)` — `"type"` found in `fieldNames`
- `TEST(TypeField, includedInDataTuple)` — `data()` variant array includes type
- `TEST(TypeField, activatableAsConserved)` — `setConserved("type")` succeeds
- `TEST(TypeField, accessibleByName)` — `getFieldIndex("type", fieldNames)` returns valid index

**Verify**: existing test suite passes unchanged (type field is unused, unallocated by default).

### Step 1.2: Type field resize and activation behavior

**Do**: No code changes — this tests the existing FieldStates machinery with the new field.

**Test** (write first):
- `TEST(TypeField, resizedWithOtherFields)` — after `setConserved("type")` and `resize(N)`,
  `type.size() == N` and `type.size() == x.size()`
- `TEST(TypeField, preservesValuesOnResize)` — set type[0..99], resize(200), verify type[0..99] unchanged

**Verify**: compile + run all existing tests.

### Step 1.3: Default type = gas in all initializers

**Do**:
- In each initializer (`sedov_init.hpp`, `noh_init.hpp`, `evrard_init.hpp`, `kelvin_helmholtz_init.hpp`,
  `turbulence_init.hpp`, `isobaric_cube_init.hpp`, `wind_shock_init.hpp`, `gresho_chan_init.hpp`,
  `file_init.hpp`): after particle resize, fill `type` with 1 (gas)
- Activate type as conserved in all propagators' `activateFields()`

**Test**:
- `TEST(TypeField, defaultTypeIsGas)` — run Sedov init, verify all `type[i] == 1`

**Verify**: run full test suite + run `sphexa --init sedov -n 20 -s 5` to confirm no crash.

### Step 1.4: Type field I/O

**Do**:
- Ensure type is included in HDF5 output when requested via `-f type`
- Ensure type can be read from HDF5 input in `file_init.hpp`

**Test**:
- `TEST(TypeFieldIO, writableToHdf5)` — write, read back, compare
- `TEST(TypeFieldIO, readableFromHdf5)` — load file with type dataset, verify values

**Verify**: run `sphexa --init sedov -n 20 -s 5 -w 1 -f x,y,z,type`, inspect output file.

### Phase 1 gate

All 9 type-field scenarios from `specs/features/type-field.md` pass.
Full existing test suite passes. No behavioral change for any existing simulation.

---

## Phase 2: Container split (DarkData + SphData + ParticleView)

**Goal**: Split `ParticlesData` into `DarkData` (basic fields) and `SphData` (SPH fields).
Create `ParticleView` for SPH kernels. Replace `get<>` with direct member access.

### Step 2.1: Create DarkData and SphData classes

**Do**:
- Create `sph/include/sph/dark_data.hpp` with basic FieldVectors, general scalars,
  data structures, I/O, and its own `dataTuple()` / `fieldNames` / `FieldStates`
- Create `sph/include/sph/sph_data.hpp` with SPH FieldVectors, SPH scalars,
  `createTables()`, and its own `dataTuple()` / `fieldNames` / `FieldStates`
- Move `numParticlesGlobal`, `numParticlesGlobalPrev`, `totalNeighbors` to `SimulationData`

**Test** (write first):
- `TEST(SimulationData, providesBasicFieldAccess)` — `simData.dark.x` compiles and is accessible
- `TEST(SimulationData, providesSphFieldAccess)` — `simData.sph.rho` compiles and is accessible
- `TEST(SimulationData, uniformFieldSizes)` — `simData.resize(100)` makes all fields size 100

**Verify**: these tests compile and pass. Old code still compiles (ParticlesData still exists).

### Step 2.2: Create ParticleView

**Do**:
- Create `sph/include/sph/particle_view.hpp` with non-owning references to all fields
  from both `DarkData` and `SphData`
- Implement `makeParticleView(SimulationData&)` factory function
- Verify that `ParticleView` satisfies the `Dataset` concept used by SPH kernels:
  has `.x`, `.y`, `.z`, `.h`, `.m`, `.rho`, `.xm`, `.nc`, `.neighbors`, etc.

**Test**:
- `TEST(ParticleView, constructsFromSimulationData)` — `makeParticleView(simData)` compiles
- `TEST(ParticleView, fieldsPointToOriginals)` — `view.x.data() == simData.dark.x.data()`

**Verify**: SPH kernel headers compile when instantiated with `ParticleView<CpuTag>`.

### Step 2.3: Update SimulationData composition

**Do**:
- Update `main/src/sphexa/simulation_data.hpp`: replace `HydroData hydro` with
  `DarkData<AccType> dark` + `SphData<AccType> sph`
- Add `SimulationData::resize()`, `setOutputFields()` iterating both containers
- Add `numParticlesGlobal`, `numParticlesGlobalPrev`, `totalNeighbors` as SimulationData members
- Split `loadOrStoreAttributes`: dark handles basic, sph handles SPH attributes

**Test**:
- `TEST(SimulationData, fieldActivationPerPropagator)` — activate different fields on each container

**Verify**: compiles.

### Step 2.4: Update propagators — field access

**Do** (one propagator at a time, starting with simplest):
1. `NbodyProp` — only uses dark fields. Replace `get<"x">(d)` → `d.dark.x`,
   `d.hydro.g` → `d.dark.g`, etc. No SPH view needed.
2. `HydroVeProp` — uses both. Build `ParticleView` for SPH kernel calls.
   Replace `get<>` in sync/halo exchange with `d.dark.*` / `d.sph.*`.
3. `HydroProp` (standard) — same pattern as VE.
4. `HydroVeBdtProp`, `TurbVeProp`, `TurbVeBdtProp` — same pattern.
5. `HydroGrackleProp`, `DiskProp` — same pattern (conditionally compiled).
6. Update `ipropagator.hpp`: `printIterationTimings` takes explicit values,
   `outputAllocatedFields` iterates `dark.data()` + `sph.data()` + `chem.data()`.
7. Update `main/src/sphexa/sphexa.cpp`: references to `simData.hydro.*` → `simData.dark.*`.

**Test** (after each propagator):
- Run existing test suite for that propagator's tests
- After all propagators done:
  - `TEST_F(SimulationDataRegression, hydroVePropSedov)` — 10-step Sedov matches pre-refactor
  - `TEST_F(SimulationDataRegression, hydroPropSedov)` — same for standard SPH
  - `TEST_F(SimulationDataRegression, nbodyPropGravityOnly)` — gravity-only works

**Verify**: full test suite passes. Run `sphexa --init sedov -n 20 -s 10` and compare output
with pre-refactor checkpoint (bitwise identical or within float tolerance).

### Step 2.5: Domain sync with two containers (MPI)

**Do**: Update domain sync calls in all propagators to pass fields from both containers
via `std::tuple_cat`.

**Test**:
- `TEST_F(SimulationDataMpi, domainSyncPreservesFields)` — 2+ ranks, sync, verify all
  field values preserved for assigned particles. Compare particle id → field values
  before and after sync.

**Verify**: MPI test suite passes.

### Step 2.6: Remove ParticlesData

**Do**: Once all code uses DarkData/SphData/ParticleView, remove or deprecate `particles_data.hpp`.
Remove `field_get.hpp` usage from propagators.

**Verify**: full build, full test suite, no references to old class remain.

### Phase 2 gate

All 9 simulation-data scenarios from `specs/features/simulation-data.md` pass.
Full existing test suite passes. Regression tests confirm identical output.

---

## Phase 3: Global type partition and gas-only layout

**Goal**: Implement Order-1 ↔ Order-2 partition, gas range finding, gas-only layout,
gas-only treeView, and gas-only halo exchange lists.

### Step 3.1: stablePartitionByType

**Do**: Implement in `domain/include/cstone/domain/type_partition.hpp`.
Stable partition: all type=0 before type=1, preserving SFC order within each group.
Store permutation vector for reverse.

**Test** (write first):
- `TEST(TypePartition, separatesTypes)` — known input, verify output order
- `TEST(TypePartition, preservesFieldValues)` — apply to x/y/z/m/type arrays, verify tuples intact
- `TEST(TypePartition, allGasUnchanged)` — all type=1, array unchanged
- `TEST(TypePartition, allDmUnchanged)` — all type=0, array unchanged

**Verify**: unit tests pass.

### Step 3.2: Reverse partition

**Do**: Implement `reversePermutation` using stored permutation vector.

**Test**:
- `TEST(TypePartition, reverseRestoresOrder1)` — partition then reverse, verify identical to input
- `TEST(TypePartition, roundTripPreservesAllFields)` — all basic + SPH fields survive round trip

**Verify**: unit tests pass.

### Step 3.3: Partition with halos

**Do**: Ensure partition operates on full array range [0, size) including halos.

**Test**:
- `TEST(TypePartition, includesHalos)` — particles at [0,start) and [end,size) are also partitioned
- `TEST(TypePartition, permutesAllArrays)` — dark + sph fields all permuted consistently

**Verify**: unit tests pass.

### Step 3.4: findAssignedRange

**Do**: Implement SFC key binary search to find gas assigned range within the gas sub-array.

**Test**:
- `TEST(GasRange, foundBySfcKeyComparison)` — known SFC keys, verify gasAssignedStart/End
- `TEST(GasRange, haloStructureValid)` — gasStart <= gasAssignedStart <= gasAssignedEnd <= gasEnd
- `TEST(GasRange, noAssignedGas)` — rank with all-DM assigned, gas range empty

**Verify**: unit tests pass.

### Step 3.5: buildSubLayout

**Do**: Implement gas-only layout by counting gas particles per leaf cell using SFC key scan.

**Test**:
- `TEST(GasLayout, computedFromSfcKeys)` — verify per-cell counts match manual count
- `TEST(GasLayout, sumMatchesGasCount)` — sum across cells == total gas particles

**Verify**: unit tests pass.

### Step 3.6: Gas-only neighbor search

**Do**: Construct gas-only `OctreeNsView` by swapping layout pointer. Call `findNeighbors`
with gas treeView.

**Test**:
- `TEST(GasLayout, enablesGasOnlyNeighborSearch)` — all returned indices are gas particles
- `TEST(GasLayout, neighborIndicesAbsolute)` — indices valid in full array, point to gas data

**Verify**: unit tests pass. This is the key correctness proof for the gas-only layout approach.

### Step 3.7: Gas-only halo exchange lists

**Do**: Derive gas SendList/RecvList from gas layout + existing halo peer information.

**Test**:
- `TEST(GasHaloExchange, derivedFromGasLayout)` — lists reference gas particles in halo cells
- `TEST_F(GasHaloExchangeMpi, transfersCorrectValues)` — MPI exchange produces correct values

**Verify**: MPI tests pass.

### Step 3.8: GPU partition

**Do**: Implement GPU variant using `thrust::stable_partition` or equivalent.

**Test**:
- `TEST(TypePartitionGpu, matchesCpuResult)` — GPU and CPU produce identical orderings

**Verify**: GPU test passes.

### Phase 3 gate

All 19 partition/layout scenarios from `specs/features/type-reindexing.md` pass.
Phase 3 is independently testable — no propagator changes needed yet.

---

## Phase 4: HydroDarkProp propagator

**Goal**: Implement the full propagator with two-pass neighbor search, gravity-first
timestep cycle, and gas-only SPH.

### Step 4.1: Acceleration accumulation change

**Do**: Change SPH momentum kernels from `ax[i] = ...` to `ax[i] += ...` (ay, az same).
Add zero ax/ay/az step before SPH in ALL existing propagators.

**Test**:
- `TEST_F(SimulationDataRegression, hydroVePropSedov)` — verify no regression (zeroing + += = same result as =)
- `TEST_F(SimulationDataRegression, hydroPropSedov)` — same

**Verify**: full test suite passes. This is a cross-cutting change that must not break anything.

### Step 4.2: HydroDarkProp skeleton

**Do**: Create `hydro_dark.hpp` with:
- `activateFields()` — activate basic fields as conserved, SPH fields as dependent
- `conservedFields()` — return list
- `sync()` — call `domain.syncGrav()` with fields from both containers
- `computeForces()` — stub that just does gravity (like NbodyProp)
- `integrate()` — same as existing (positions, velocities, smoothing length)
- Register in factory as `"hydro-dark"`

**Test**:
- `TEST(PropagatorFactory, hydroDarkRegistered)` — factory returns HydroDarkProp
- `TEST(PropagatorFactory, cliSelectsHydroDark)` — `--prop hydro-dark` works
- Run with all-gas particles: should behave like NbodyProp (gravity only, no SPH yet)

**Verify**: compiles, runs, gravity works.

### Step 4.3: Two-pass neighbor search

**Do**: Add pass 1 (Order-1, all types, h adjustment) and pass 2 placeholder.

**Test**:
- `TEST_F(HydroDarkProp, pass1SetsHForAll)` — h adjusted for all particles after pass 1

**Verify**: runs without crash.

### Step 4.4: Partition integration

**Do**: Add partition → gas range → gas layout → gas treeView → reverse cycle.
Call Phase 3 functions from `computeForces()`.

**Test**:
- `TEST_F(HydroDarkProp, partitionBeforeSph)` — after partition, verify Order-2
- `TEST_F(HydroDarkProp, reverseBeforeIntegrate)` — after reverse, verify Order-1
- `TEST_F(HydroDarkProp, typePreservedAcrossTimestep)` — types unchanged after full step

**Verify**: runs, partitions, reverses, integrates.

### Step 4.5: Gas-only SPH

**Do**: Add pass 2 neighbor search (gas layout) + full VE SPH kernel chain with gas-only
halo exchanges. Use `ParticleView` for kernel calls. Only operate on `[gasAssignedStart, gasAssignedEnd)`.

**Test**:
- `TEST_F(HydroDarkProp, pass2GasOnlyNeighbors)` — neighbor list gas-only
- `TEST_F(HydroDarkProp, sphNeighborsGasOnly)` — no DM index in neighbor list
- `TEST_F(HydroDarkProp, dmHUnchangedByPass2)` — DM h from pass 1 preserved
- `TEST_F(HydroDarkProp, sphFieldsUnchangedForDm)` — DM SPH fields untouched

**Verify**: SPH runs on gas sub-array only.

### Step 4.6: Force separation verification

**Do**: No code changes — verification tests.

**Test**:
- `TEST_F(HydroDarkProp, dmGravityOnly)` — DM ax = gravity only
- `TEST_F(HydroDarkProp, gravityBothTypes)` — two-body 1/r^2 test
- `TEST_F(HydroDarkProp, accelerationAccumulation)` — gas ax = gravity + SPH
- `TEST_F(HydroDarkProp, pureGasMatchesHydroVeProp)` — all-gas HydroDarkProp == HydroVeProp

**Verify**: physics correctness established. The pure-gas comparison is the strongest proof
that the partition + gas layout + two-pass approach produces identical results.

### Step 4.7: MPI correctness

**Do**: No code changes — MPI-specific tests.

**Test**:
- `TEST_F(HydroDarkPropMpi, syncPreservesTypes)` — types survive domain redistribution
- `TEST_F(HydroDarkPropMpi, particleCountConserved)` — DM + gas counts unchanged
- `TEST_F(HydroDarkPropMpi, haloExchangeIncludesType)` — halo types correct
- `TEST_F(HydroDarkPropMpi, midSphHaloExchange)` — gas halo exchange works
- `TEST_F(HydroDarkPropMpi, allFourHaloExchanges)` — all mid-SPH exchanges correct

**Verify**: MPI tests pass with 2+ ranks.

### Step 4.8: Integration and timestep

**Do**: Verify integration and timestep computation.

**Test**:
- `TEST_F(HydroDarkProp, allParticlesIntegrated)` — both types updated
- `TEST_F(HydroDarkProp, timestepBothConstraints)` — dt satisfies gravity + Courant
- `TEST_F(HydroDarkProp, dmTrajectoryGravityOnly)` — analytical orbit comparison

**Verify**: multi-step runs are stable.

### Step 4.9: Conservation laws

**Do**: No code changes — multi-step conservation tests.

**Test**:
- `TEST_F(HydroDarkPropConservation, totalEnergy)` — 100 steps, energy conserved
- `TEST_F(HydroDarkPropConservation, totalMomentum)` — 100 steps, momentum conserved
- `TEST_F(HydroDarkPropConservation, totalMass)` — mass unchanged

**Verify**: conservation within expected floating-point tolerance.

### Step 4.10: BDT variant

**Do**: Implement `HydroDarkBdtProp` — full sync rebuilds partition, partial sync reuses.

**Test**:
- `TEST_F(HydroDarkBdtProp, fullSyncRebuilds)` — partition rebuilt on substep 0
- `TEST_F(HydroDarkBdtProp, partialSyncReusesPartition)` — partition reused on substep > 0

**Verify**: BDT runs without crash on GPU.

### Step 4.11: GPU parity

**Do**: Verify GPU kernels produce same results.

**Test**:
- `TEST_F(HydroDarkPropGpu, forcesMatchCpu)` — GPU vs CPU acceleration comparison
- `TEST_F(HydroDarkPropGpu, partitionMatchesCpu)` — GPU vs CPU partition comparison

**Verify**: GPU tests pass.

### Phase 4 gate

All 30 HydroDarkProp scenarios from `specs/features/hydro-dark-prop.md` pass.
Pure-gas comparison proves correctness of the architecture.
Conservation laws prove physical correctness.
MPI tests prove distributed correctness.

---

## Phase 5: Mixed initial conditions

**Goal**: Support loading mixed DM + gas particles from files.

### Step 5.1: Type column in file I/O

**Do**:
- `file_init.hpp`: read `type` dataset from HDF5 if present, default to 1 (gas) if absent
- ASCII reader: parse type column if present

**Test** (write first):
- `TEST_F(MixedInit, hdf5WithTypeColumn)` — load file with types, verify correct
- `TEST_F(MixedInit, hdf5WithoutTypeDefaultsGas)` — legacy file → all gas
- `TEST_F(MixedInit, asciiWithTypeColumn)` — ASCII with types works

**Verify**: tests pass.

### Step 5.2: Built-in initializer compatibility

**Do**: Verify all built-in initializers set type = 1 (already done in Phase 1).

**Test**:
- `TEST_F(MixedInit, builtInInitSetGas)` — Sedov, Noh, Evrard all produce type=1

**Verify**: tests pass.

### Step 5.3: Output and checkpoint

**Do**: Ensure type field in output and checkpoint/restart.

**Test**:
- `TEST_F(MixedInitIO, typeFieldInOutput)` — type appears in HDF5 output
- `TEST_F(MixedInitIO, checkpointRestoreTypes)` — write checkpoint, restart, types match

**Verify**: tests pass.

### Step 5.4: Type count reporting

**Do**: Add DM/gas count reporting at simulation startup.

**Test**:
- `TEST_F(MixedInit, typeCountsReported)` — verify count output

**Verify**: run `sphexa --init file:mixed.h5 --prop hydro-dark -s 1` and check output.

### Step 5.5: Integration test

**Do**: Create a test HDF5 file with gas cloud + DM halo.

**Test**:
- `TEST_F(MixedInit, evrardWithDmHalo)` — gas collapses, DM orbits, 10 steps stable

**Verify**: integration test passes. This is the end-to-end proof.

### Phase 5 gate

All 8 mixed-init scenarios from `specs/features/mixed-init.md` pass.

---

## Correctness verification summary

| Phase | Key correctness proof | What it demonstrates |
|-------|----------------------|---------------------|
| 1 | Existing test suite unchanged | Type field is additive, no behavioral change |
| 2 | Regression: HydroVeProp Sedov matches pre-refactor | Container split preserves all physics |
| 3 | Gas-only neighbor search returns only gas indices | Partition + layout approach is sound |
| 4 | Pure-gas HydroDarkProp == HydroVeProp | Full architecture produces identical results |
| 4 | Conservation laws (energy, momentum, mass) | Physics is correct for mixed types |
| 4 | DM gravity-only analytical orbit | DM force separation correct |
| 5 | Evrard + DM halo end-to-end | Full system works with real physics |

Each phase is gated: all feature scenarios pass before proceeding to the next phase.
Full existing test suite runs after every step to catch regressions.

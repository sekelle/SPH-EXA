# Test Map: Spec → GTest

Maps each specification feature to concrete test files, suites, and fixtures.
Follows existing SPH-EXA test conventions: GTest with `TEST()` / `TEST_F()`,
template helpers for type parametrization, MPI fixtures, GPU `.cu` files.

## Test file organization

```
sph/test/
├── type_field.cpp              — Phase 1: type field unit tests
├── simulation_data.cpp         — Phase 2: SimulationData split tests

domain/test/unit/
├── domain/type_partition.cpp   — Phase 3: partition + gas layout unit tests

domain/test/unit_cuda/
├── domain/type_partition_gpu.cu — Phase 3: GPU partition tests

main/test/
├── hydro_dark_prop.cpp         — Phase 4: propagator unit tests
├── mixed_init.cpp              — Phase 5: initialization tests

domain/test/integration_mpi/
├── domain_types.cpp            — Phase 1+3+4: MPI tests for type field sync + gas halo exchange

main/test/
├── hydro_dark_conservation.cpp — Phase 4: conservation law tests (multi-step)
```

## Phase 1: Type Field

| Spec scenario | Test suite | Test name | File |
|---|---|---|---|
| Type field exists in field registry | TypeField | existsInFieldRegistry | sph/test/type_field.cpp |
| Type field included in dataTuple | TypeField | includedInDataTuple | sph/test/type_field.cpp |
| Type field resized with other fields | TypeField | resizedWithOtherFields | sph/test/type_field.cpp |
| Type field preserves values on resize | TypeField | preservesValuesOnResize | sph/test/type_field.cpp |
| Default type is gas | TypeField | defaultTypeIsGas | sph/test/type_field.cpp |
| Type field activatable as conserved | TypeField | activatableAsConserved | sph/test/type_field.cpp |
| Type field accessible by name | TypeField | accessibleByName | sph/test/type_field.cpp |
| Type field writable to HDF5 | TypeFieldIO | writableToHdf5 | main/test/mixed_init.cpp |
| Type field readable from HDF5 | TypeFieldIO | readableFromHdf5 | main/test/mixed_init.cpp |

## Phase 2: SimulationData Split

| Spec scenario | Test suite | Test name | File |
|---|---|---|---|
| Basic field access | SimulationData | providesBasicFieldAccess | sph/test/simulation_data.cpp |
| SPH field access | SimulationData | providesSphFieldAccess | sph/test/simulation_data.cpp |
| Uniform field sizes | SimulationData | uniformFieldSizes | sph/test/simulation_data.cpp |
| Propagator template parameter | SimulationData | propagatorTemplateParameter | sph/test/simulation_data.cpp |
| HydroVeProp regression | SimulationDataRegression | hydroVePropSedov | main/test/hydro_dark_prop.cpp |
| HydroProp regression | SimulationDataRegression | hydroPropSedov | main/test/hydro_dark_prop.cpp |
| NbodyProp regression | SimulationDataRegression | nbodyPropGravityOnly | main/test/hydro_dark_prop.cpp |
| Domain sync with new structure | SimulationDataMpi | domainSyncPreservesFields | domain/test/integration_mpi/domain_types.cpp |
| Field activation per propagator | SimulationData | fieldActivationPerPropagator | sph/test/simulation_data.cpp |

## Phase 3: Global Type Partition + Gas Layout

| Spec scenario | Test suite | Test name | File |
|---|---|---|---|
| Stable partition separates types | TypePartition | separatesTypes | domain/test/unit/domain/type_partition.cpp |
| Partition includes halos | TypePartition | includesHalos | domain/test/unit/domain/type_partition.cpp |
| Partition preserves field values | TypePartition | preservesFieldValues | domain/test/unit/domain/type_partition.cpp |
| Reverse restores Order-1 | TypePartition | reverseRestoresOrder1 | domain/test/unit/domain/type_partition.cpp |
| Round-trip preserves all fields | TypePartition | roundTripPreservesAllFields | domain/test/unit/domain/type_partition.cpp |
| Permutes all arrays | TypePartition | permutesAllArrays | domain/test/unit/domain/type_partition.cpp |
| All-gas unchanged | TypePartition | allGasUnchanged | domain/test/unit/domain/type_partition.cpp |
| All-DM unchanged | TypePartition | allDmUnchanged | domain/test/unit/domain/type_partition.cpp |
| GPU matches CPU | TypePartitionGpu | matchesCpuResult | domain/test/unit_cuda/domain/type_partition_gpu.cu |
| Gas range by SFC key | GasRange | foundBySfcKeyComparison | domain/test/unit/domain/type_partition.cpp |
| Gas halo structure valid | GasRange | haloStructureValid | domain/test/unit/domain/type_partition.cpp |
| No assigned gas | GasRange | noAssignedGas | domain/test/unit/domain/type_partition.cpp |
| Gas layout from SFC keys | GasLayout | computedFromSfcKeys | domain/test/unit/domain/type_partition.cpp |
| Gas layout sum matches count | GasLayout | sumMatchesGasCount | domain/test/unit/domain/type_partition.cpp |
| Gas-only neighbor search | GasLayout | enablesGasOnlyNeighborSearch | domain/test/unit/domain/type_partition.cpp |
| Neighbor indices absolute | GasLayout | neighborIndicesAbsolute | domain/test/unit/domain/type_partition.cpp |
| Gas halo lists from layout | GasHaloExchange | derivedFromGasLayout | domain/test/unit/domain/type_partition.cpp |
| Gas halo exchange correct (MPI) | GasHaloExchangeMpi | transfersCorrectValues | domain/test/integration_mpi/domain_types.cpp |
| Partition linear scaling | TypePartition | linearScaling | domain/test/performance/ |

## Phase 4: HydroDarkProp

| Spec scenario | Test suite | Test name | File |
|---|---|---|---|
| DM gravity only | HydroDarkProp | dmGravityOnly | main/test/hydro_dark_prop.cpp |
| Pure gas matches HydroVeProp | HydroDarkProp | pureGasMatchesHydroVeProp | main/test/hydro_dark_prop.cpp |
| Gravity both types | HydroDarkProp | gravityBothTypes | main/test/hydro_dark_prop.cpp |
| Acceleration accumulation | HydroDarkProp | accelerationAccumulation | main/test/hydro_dark_prop.cpp |
| SPH neighbors gas only | HydroDarkProp | sphNeighborsGasOnly | main/test/hydro_dark_prop.cpp |
| SPH fields unchanged for DM | HydroDarkProp | sphFieldsUnchangedForDm | main/test/hydro_dark_prop.cpp |
| Pass 1 sets h for all | HydroDarkProp | pass1SetsHForAll | main/test/hydro_dark_prop.cpp |
| Pass 2 gas-only neighbors | HydroDarkProp | pass2GasOnlyNeighbors | main/test/hydro_dark_prop.cpp |
| DM h unchanged by pass 2 | HydroDarkProp | dmHUnchangedByPass2 | main/test/hydro_dark_prop.cpp |
| Mid-SPH halo exchange (MPI) | HydroDarkPropMpi | midSphHaloExchange | domain/test/integration_mpi/domain_types.cpp |
| Four halo exchanges (MPI) | HydroDarkPropMpi | allFourHaloExchanges | domain/test/integration_mpi/domain_types.cpp |
| Partition before SPH | HydroDarkProp | partitionBeforeSph | main/test/hydro_dark_prop.cpp |
| Reverse before integrate | HydroDarkProp | reverseBeforeIntegrate | main/test/hydro_dark_prop.cpp |
| Type preserved across timestep | HydroDarkProp | typePreservedAcrossTimestep | main/test/hydro_dark_prop.cpp |
| All particles integrated | HydroDarkProp | allParticlesIntegrated | main/test/hydro_dark_prop.cpp |
| DM trajectory gravity only | HydroDarkProp | dmTrajectoryGravityOnly | main/test/hydro_dark_prop.cpp |
| Timestep both constraints | HydroDarkProp | timestepBothConstraints | main/test/hydro_dark_prop.cpp |
| Sync preserves types (MPI) | HydroDarkPropMpi | syncPreservesTypes | domain/test/integration_mpi/domain_types.cpp |
| Particle count conserved (MPI) | HydroDarkPropMpi | particleCountConserved | domain/test/integration_mpi/domain_types.cpp |
| Halo exchange includes type (MPI) | HydroDarkPropMpi | haloExchangeIncludesType | domain/test/integration_mpi/domain_types.cpp |
| BDT full sync rebuilds | HydroDarkBdtProp | fullSyncRebuilds | main/test/hydro_dark_prop.cpp |
| BDT partial sync reuses | HydroDarkBdtProp | partialSyncReusesPartition | main/test/hydro_dark_prop.cpp |
| Total energy conserved | HydroDarkPropConservation | totalEnergy | main/test/hydro_dark_conservation.cpp |
| Total momentum conserved | HydroDarkPropConservation | totalMomentum | main/test/hydro_dark_conservation.cpp |
| Total mass conserved | HydroDarkPropConservation | totalMass | main/test/hydro_dark_conservation.cpp |
| Factory registration | PropagatorFactory | hydroDarkRegistered | main/test/hydro_dark_prop.cpp |
| CLI selection | PropagatorFactory | cliSelectsHydroDark | main/test/hydro_dark_prop.cpp |
| GPU forces match CPU | HydroDarkPropGpu | forcesMatchCpu | main/test/ (CUDA) |
| GPU partition matches CPU | HydroDarkPropGpu | partitionMatchesCpu | main/test/ (CUDA) |

## Phase 5: Mixed Init

| Spec scenario | Test suite | Test name | File |
|---|---|---|---|
| HDF5 with type column | MixedInit | hdf5WithTypeColumn | main/test/mixed_init.cpp |
| HDF5 without type defaults gas | MixedInit | hdf5WithoutTypeDefaultsGas | main/test/mixed_init.cpp |
| ASCII with type column | MixedInit | asciiWithTypeColumn | main/test/mixed_init.cpp |
| Type counts reported | MixedInit | typeCountsReported | main/test/mixed_init.cpp |
| Built-in init sets gas | MixedInit | builtInInitSetGas | main/test/mixed_init.cpp |
| Type field in output | MixedInitIO | typeFieldInOutput | main/test/mixed_init.cpp |
| Checkpoint restore types | MixedInitIO | checkpointRestoreTypes | main/test/mixed_init.cpp |
| Evrard with DM halo | MixedInit | evrardWithDmHalo | main/test/mixed_init.cpp |

## Total test count

| Phase | Unit | MPI | GPU | Total |
|-------|------|-----|-----|-------|
| 1. Type field | 7 | — | — | 7 |
| 2. SimData split | 6 | 1 | — | 7 |
| 3. Partition + gas layout | 14 | 1 | 1 | 16 |
| 4. HydroDarkProp | 16 | 5 | 2 | 23 |
| 5. Mixed init | 7 | — | — | 7 |
| **Total** | **50** | **7** | **3** | **60** |

Plus 1 performance test (Phase 3) and 1 integration test (Phase 5) = **62 scenarios**.

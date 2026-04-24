# Feature: Global Type Partition and Gas-Only Layout (Phase 3)

Implement the Order-1 ↔ Order-2 global stable partition that separates particles by type,
and build a gas-only layout for SPH computation.

## Definitions

- **Order-1**: particles sorted by SFC key (existing ordering, mixed types)
- **Order-2**: particles partitioned by type (DM first, gas second), SFC order preserved
  within each type group. This is a global partition, not per-cell.
- **Gas sub-array**: the contiguous gas portion of the Order-2 array, itself SFC-sorted
  with its own halo structure (gasHaloFront | gasAssigned | gasHaloBack)
- **Gas-only layout**: maps octree leaf cells to gas particle ranges, computed by counting
  gas particles per cell using their SFC keys
- **Gas-only treeView**: copy of OctreeNsView with layout pointer swapped to gas-only layout
- **Gas-only halo lists**: SendList/RecvList for gas particles, derived from gas layout +
  existing halo peer information (local computation, no MPI)

## Scenarios — Partition

### Scenario: Stable partition separates types
Given: particles in Order-1 with types [1,0,1,0,1,1,0,0,1,0] and SFC keys [k0..k9]
When: stable partition by type is applied
Then: result is [0,0,0,0, 1,1,1,1,1,1] (all DM first, all gas second)
And: within each group, SFC key order is preserved
Test: `TEST(TypePartition, separatesTypes)`

### Scenario: Partition includes halo particles
Given: particles with halos at [0, startIndex) and [endIndex, size)
When: stable partition is applied to ALL particles (assigned + halos)
Then: both assigned and halo particles are correctly partitioned
And: no particle is lost or duplicated
Test: `TEST(TypePartition, includesHalos)`

### Scenario: Partition preserves all field values
Given: particles with known positions, velocities, masses, types, and SPH fields
When: stable partition is applied (all arrays permuted together)
Then: each particle's (id, x, y, z, vx, vy, vz, m, type, rho, u, ...) tuple is unchanged
Test: `TEST(TypePartition, preservesFieldValues)` (INV-4)

### Scenario: Reverse partition restores Order-1
Given: particles partitioned from Order-1 to Order-2 with stored permutation
When: the inverse permutation is applied
Then: particle ordering matches the original Order-1 exactly
Test: `TEST(TypePartition, reverseRestoresOrder1)` (INV-7)

### Scenario: Round-trip preserves all fields
Given: particles with all fields (basic + SPH) populated with known values
When: partition to Order-2 then reverse to Order-1
Then: all field values are identical to the original (bitwise for integers, exact for floats)
Test: `TEST(TypePartition, roundTripPreservesAllFields)`

### Scenario: Partition permutes ALL field arrays
Given: particles with basic and SPH fields populated
When: partition is applied
Then: ALL active FieldVectors (basic + SPH) are permuted consistently
And: for any particle index i in Order-2, all field arrays refer to the same particle
Test: `TEST(TypePartition, permutesAllArrays)`

### Scenario: Partition handles all-gas (no DM)
Given: particles where all types are 1 (gas)
When: partition is applied
Then: array is unchanged (already partitioned)
Test: `TEST(TypePartition, allGasUnchanged)`

### Scenario: Partition handles all-DM (no gas)
Given: particles where all types are 0 (DM)
When: partition is applied
Then: array is unchanged (already partitioned)
And: gas sub-array is empty
Test: `TEST(TypePartition, allDmUnchanged)`

### Scenario: Partition works with GPU arrays
Given: particles stored in DeviceVector (GPU memory)
When: stable partition is applied on GPU
Then: results match CPU partition
Test: `TEST(TypePartitionGpu, matchesCpuResult)` (CUDA)

## Scenarios — Gas assigned range

### Scenario: Gas assigned range found by SFC key comparison
Given: particles in Order-2 with domain SFC boundaries [keyStart, keyEnd)
When: gas assigned range is computed by binary search on gas particle SFC keys
Then: gasAssignedStart is the first gas particle with key >= keyStart
And: gasAssignedEnd is the first gas particle with key >= keyEnd
Test: `TEST(GasRange, foundBySfcKeyComparison)`

### Scenario: Gas halo structure is valid
Given: particles in Order-2
When: gas range is computed
Then: gasStart <= gasAssignedStart <= gasAssignedEnd <= gasEnd
And: gas particles in [gasStart, gasAssignedStart) are front halos
And: gas particles in [gasAssignedEnd, gasEnd) are back halos
Test: `TEST(GasRange, haloStructureValid)`

### Scenario: Gas range handles rank with no gas
Given: a rank whose assigned particles are all DM
When: gas range is computed
Then: gasAssignedStart == gasAssignedEnd (empty assigned range)
And: gas halos may still exist
Test: `TEST(GasRange, noAssignedGas)`

## Scenarios — Gas-only layout

### Scenario: Gas-only layout computed from gas SFC keys
Given: gas particles in Order-2 with known SFC keys and tree leaf boundaries
When: gas-only layout is computed by counting gas particles per cell
Then: for each cell i, gasLayout[i+1] - gasLayout[i] equals the number of gas particles in that cell
Test: `TEST(GasLayout, computedFromSfcKeys)` (INV-8)

### Scenario: Gas layout sums match gas particle count
Given: gas-only layout for all leaf cells
When: summed across all cells
Then: total equals the number of gas particles (assigned + halos)
Test: `TEST(GasLayout, sumMatchesGasCount)`

### Scenario: Gas treeView enables gas-only neighbor search
Given: a gas-only OctreeNsView (same tree structure, gas layout pointer)
When: findNeighbors is called with gas treeView and gas particle x/y/z/h
Then: returned neighbor indices are all gas particles (absolute array indices)
And: no DM particle index appears in the neighbor list
Test: `TEST(GasLayout, enablesGasOnlyNeighborSearch)` (INV-9)

### Scenario: Gas-only neighbor indices are absolute
Given: gas-only neighbor search results
When: indices are used to access x[], y[], z[], m[] arrays
Then: they point to valid gas particle data in the full particle array
Test: `TEST(GasLayout, neighborIndicesAbsolute)`

## Scenarios — Gas-only halo exchange

### Scenario: Gas halo lists derived from gas layout
Given: gas-only layout and existing halo peer information
When: gas-only SendList/RecvList are computed
Then: send lists reference gas particles in cells that peers need as halos
And: recv lists reference positions for incoming gas halo data
Test: `TEST(GasHaloExchange, derivedFromGasLayout)`

### Scenario: Gas halo exchange transfers correct field values
Given: gas particles with computed SPH field values (xm, divv, etc.)
When: gas-only halo exchange is performed
Then: gas halo particles receive the correct field values from owning ranks
Test: `TEST_F(GasHaloExchangeMpi, transfersCorrectValues)` (MPI, 2+ ranks)

### Scenario: Gas halo exchange is local computation (no MPI discovery)
Given: existing halo peer relationships from domain.sync()
When: gas halo lists are built
Then: no MPI communication is needed (only local remapping)
Test: `TEST(GasHaloExchange, noMpiDiscovery)` (verified by construction)

## Scenarios — Performance

### Scenario: Partition scales linearly
Given: N particles
When: stable partition is applied
Then: the operation completes in O(N) time
Test: `TEST(TypePartition, linearScaling)` (performance)

## Invariants tested
- INV-4 (identity preserved across partition)
- INV-7 (round-trip identity)
- INV-8 (layout consistency)
- INV-9 (DM excluded from SPH neighbor lists)

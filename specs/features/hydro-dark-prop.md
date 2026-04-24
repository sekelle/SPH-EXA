# Feature: HydroDarkProp Propagator (Phase 4)

New propagator that handles mixed dark matter + gas particle simulations. Computes gravity
for all particles (Order-1) and SPH forces for gas particles only (Order-2), using a global
stable partition to switch between orderings.

## Timestep cycle

```
 1. domain.syncGrav()                — Order-1, all particles
 2. fill halos                       — Order-1, basic + SPH fields
 3. findNeighborsSph pass 1          — Order-1, all types, adjusts h for all particles
 4. zero ax, ay, az                  — all particles
 5. gravity (all particles)          — Order-1, ax/ay/az += gravity
 6. stable partition → Order-2       — all particles (assigned+halos)
 7. find gas assigned range          — SFC key comparison with domain boundaries
 8. build gas-only layout + treeView — recount per cell, swap layout pointer
 9. build gas-only halo lists        — local remap, no MPI
10. findNeighborsSph pass 2          — Order-2, gas only, gas-only neighbors, adjusts gas h
11. SPH kernels (gas only)           — with gas-only halo exchanges between sub-steps
                                       ax/ay/az += SPH forces, du = SPH energy rate
12. reverse partition → Order-1      — restore from stored permutation
13. integrate (all particles)        — Order-1, positions/velocities
```

## Scenarios — Force computation

### Scenario: DM particles receive only gravitational acceleration
Given: a simulation with 50 DM + 50 gas particles in a gravitating configuration
When: one timestep of HydroDarkProp completes
Then: DM particles have ax,ay,az from gravity only (no SPH contribution)
And: gas particles have ax,ay,az from gravity + SPH
Test: `TEST_F(HydroDarkProp, dmGravityOnly)` (INV-9, INV-10, INV-11)

### Scenario: Gas SPH forces match pure-gas simulation
Given: 100 gas particles in a Sedov-like configuration (no DM)
When: HydroDarkProp computes forces for 10 steps
Then: gas accelerations match HydroVeProp results within floating-point tolerance
Test: `TEST_F(HydroDarkProp, pureGasMatchesHydroVeProp)`

### Scenario: Gravity computed for all particle types
Given: a two-body problem with one DM and one gas particle
When: gravity is computed
Then: both particles have gravitational accelerations toward each other
And: acceleration magnitude matches analytical 1/r^2
Test: `TEST_F(HydroDarkProp, gravityBothTypes)`

### Scenario: Acceleration accumulation is correct
Given: gas particles in a gravitating SPH configuration
When: forces are computed (gravity first, then SPH adds)
Then: ax[i] = gravity_contribution + sph_contribution for each gas particle
And: only ax/ay/az use += accumulation; du uses = assignment
Test: `TEST_F(HydroDarkProp, accelerationAccumulation)` (INV-15)

### Scenario: SPH neighbor lists contain only gas particles
Given: a mixed DM + gas configuration
When: gas-only neighbor search is performed in Order-2 with gas layout
Then: all neighbor indices refer to gas particles (type=1)
And: indices are absolute positions in the full particle array
Test: `TEST_F(HydroDarkProp, sphNeighborsGasOnly)` (INV-9)

### Scenario: SPH fields unchanged for DM particles
Given: DM particle SPH fields initialized to known sentinel values
When: one full timestep completes
Then: DM particle SPH fields still contain the sentinel values
Test: `TEST_F(HydroDarkProp, sphFieldsUnchangedForDm)` (INV-14)

## Scenarios — Two-pass neighbor search

### Scenario: Pass 1 sets h for all particles
Given: particles in Order-1 with full layout
When: findNeighborsSph pass 1 runs
Then: h is adjusted for ALL particles (DM + gas) based on all-type neighbor counts
And: DM h values are final (used for gravity softening)
Test: `TEST_F(HydroDarkProp, pass1SetsHForAll)`

### Scenario: Pass 2 finds gas-only neighbors and adjusts gas h
Given: particles in Order-2 with gas layout
When: findNeighborsSph pass 2 runs
Then: neighbor lists contain only gas particle indices
And: gas h is re-adjusted based on gas-only neighbor counts (overwrites pass 1 gas h)
Test: `TEST_F(HydroDarkProp, pass2GasOnlyNeighbors)`

### Scenario: DM h unchanged by pass 2
Given: DM h values set by pass 1
When: pass 2 (gas-only) runs
Then: DM h values are unchanged (pass 2 only touches gas particles)
Test: `TEST_F(HydroDarkProp, dmHUnchangedByPass2)`

## Scenarios — Gas-only halo exchange

### Scenario: Mid-SPH halo exchange works in Order-2
Given: gas SPH field values computed (xm, divv, c, prho, etc.)
When: gas-only halo exchange is performed between SPH sub-steps
Then: gas halo particles receive correct updated field values
Test: `TEST_F(HydroDarkPropMpi, midSphHaloExchange)` (MPI, 2+ ranks)

### Scenario: Four halo exchanges complete correctly
Given: VE SPH computation with exchanges after xmass, VE, IAD, and EOS
When: all four gas-only halo exchanges complete
Then: gas halo particles have correct xm, vx/vy/vz/kx, c11-c33/divv/c, prho/alpha values
Test: `TEST_F(HydroDarkPropMpi, allFourHaloExchanges)` (MPI, 2+ ranks)

## Scenarios — Ordering

### Scenario: Partition to Order-2 before SPH
Given: particles in Order-1 after gravity
When: stable partition is applied
Then: all DM particles precede all gas particles
And: SFC order preserved within each type group
Test: `TEST_F(HydroDarkProp, partitionBeforeSph)` (INV-6)

### Scenario: Reverse partition to Order-1 before integrate
Given: particles in Order-2 after SPH computation
When: reverse partition is applied
Then: particles are back in SFC order (all types interleaved)
And: all field values are consistent (gravity + SPH accelerations preserved)
Test: `TEST_F(HydroDarkProp, reverseBeforeIntegrate)` (INV-5, INV-7)

### Scenario: Type field survives full timestep cycle
Given: particles with known types
When: a full timestep (sync → pass1 → gravity → partition → pass2 → SPH → reverse → integrate) completes
Then: every particle retains its original type value
Test: `TEST_F(HydroDarkProp, typePreservedAcrossTimestep)` (INV-2)

## Scenarios — Integration

### Scenario: All particles integrated
Given: mixed DM + gas particles with computed accelerations
When: integrate is called in Order-1
Then: all particles (both types) have updated positions and velocities
Test: `TEST_F(HydroDarkProp, allParticlesIntegrated)`

### Scenario: DM particle trajectories are gravity-only
Given: a known gravitational configuration (e.g., circular orbit)
When: 100 timesteps are computed
Then: DM particle trajectories match the analytical gravity-only solution
Test: `TEST_F(HydroDarkProp, dmTrajectoryGravityOnly)`

### Scenario: Timestep respects both Courant and gravity constraints
Given: mixed particles with different characteristic timescales
When: the global timestep is computed
Then: dt satisfies both the gravitational and SPH Courant constraints
Test: `TEST_F(HydroDarkProp, timestepBothConstraints)`

## Scenarios — Domain sync with types

### Scenario: Domain sync preserves types across ranks
Given: mixed particles distributed across MPI ranks
When: domain.syncGrav() redistributes particles
Then: each particle retains its type (INV-16)
And: halo particles have correct types (INV-17)
Test: `TEST_F(HydroDarkPropMpi, syncPreservesTypes)` (MPI, 2+ ranks)

### Scenario: Particle count conserved across sync
Given: N_dm DM + N_gas gas particles across all ranks
When: domain sync redistributes particles
Then: global sum of DM particles == N_dm, gas == N_gas
Test: `TEST_F(HydroDarkPropMpi, particleCountConserved)` (MPI, 2+ ranks)

### Scenario: Halo exchange includes type field
Given: particles requiring halo exchange
When: halos are exchanged
Then: received halo particles have correct type values
Test: `TEST_F(HydroDarkPropMpi, haloExchangeIncludesType)` (MPI, 2+ ranks)

## Scenarios — Block-timestep (BDT)

### Scenario: Full sync rebuilds partition
Given: BDT propagator at substep 0 (full sync)
When: syncGrav + partition + gas layout are computed
Then: partition reflects new particle distribution
And: gas layout and gas halo lists are rebuilt
Test: `TEST_F(HydroDarkBdtProp, fullSyncRebuilds)`

### Scenario: Partial sync reuses partition
Given: BDT propagator at substep > 0 (partial sync)
When: positions change but SFC keys are unchanged
Then: the existing partition is reused (no re-sort)
And: gas layout and gas halo lists remain valid
Test: `TEST_F(HydroDarkBdtProp, partialSyncReusesPartition)`

## Scenarios — Conservation

### Scenario: Total energy conserved
Given: an isolated DM + gas system
When: 100 timesteps are computed
Then: total energy (kinetic + internal + gravitational) is conserved within tolerance
Test: `TEST_F(HydroDarkPropConservation, totalEnergy)`

### Scenario: Total momentum conserved
Given: an isolated DM + gas system
When: 100 timesteps are computed
Then: total linear momentum is conserved within tolerance
Test: `TEST_F(HydroDarkPropConservation, totalMomentum)`

### Scenario: Mass conserved
Given: mixed particles
When: 100 timesteps are computed
Then: sum of all particle masses is unchanged (INV-3)
Test: `TEST_F(HydroDarkPropConservation, totalMass)`

## Scenarios — Factory and CLI

### Scenario: Propagator registered in factory
Given: the propagator factory
When: `"hydro-dark"` is requested
Then: a HydroDarkProp instance is returned
Test: `TEST(PropagatorFactory, hydroDarkRegistered)`

### Scenario: CLI selects HydroDarkProp
Given: command line arguments `--prop hydro-dark`
When: the propagator is created
Then: a HydroDarkProp is instantiated
Test: `TEST(PropagatorFactory, cliSelectsHydroDark)`

## Scenarios — GPU parity

### Scenario: GPU forces match CPU forces
Given: identical mixed particle configurations on CPU and GPU
When: one timestep is computed on each
Then: accelerations match within GPU floating-point tolerance
Test: `TEST_F(HydroDarkPropGpu, forcesMatchCpu)` (CUDA)

### Scenario: GPU partition matches CPU
Given: identical particle data on CPU and GPU
When: stable partition is applied on each
Then: resulting orderings are identical
Test: `TEST_F(HydroDarkPropGpu, partitionMatchesCpu)` (CUDA)

## Invariants tested
- INV-1, INV-2, INV-3 (conservation)
- INV-5, INV-6 (ordering)
- INV-7 (partition round-trip)
- INV-9, INV-10, INV-11 (force type-physics separation)
- INV-14, INV-15 (field integrity)
- INV-16, INV-17 (type through domain sync)

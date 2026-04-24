# Implementation of particle types in SPH-EXA

## Overview
The goal is to implement different types of particles in SPH-EXA. Examples of different types are
gas particles, dark matter particles or star particles. Currently, all particles are treated as gas particles.
Different types of particles are subject to different types of forces.

## Design principles

### Types of forces
We support:
  - gravitational forces
  - hydrodynamical (SPH) forces

### Types of fields
Fields are the variables associated with a particle.

* Basic fields: all particles require them for time-step integration. They are:
    - keys
    - x
    - y
    - z
    - x_m1
    - y_m1
    - z_m1
    - m
    - h
    - vx
    - vy
    - vz
    - ax
    - ay
    - az
    - rung
    - id
    - type (new field, uint8_t, supports 256 different types)

* SPH fields: required by gas particles subject to hydrodynamical forces:
    - rho
    - temp
    - u
    - prho
    - tdpdTrho
    - c
    - cv
    - mue
    - mui
    - divv
    - curlv
    - c11, c12, c13, c22, c23, c33
    - alpha
    - xm
    - kx
    - gradh
    - dV11, dV12, dV13, dV22, dV23, dV33

These are the sets of possible fields that must be supported. However, the specific Propagator class
decides which of these fields it will activate and use.

### Type values
  - 0: dark matter (DM) — gravity only
  - 1: gas — gravity + SPH

### Desired particle types

* Dark matter particles: they only have the basic fields, subject to gravity only
* Gas particles: they have the basic plus the SPH fields, subject to gravity and SPH

Extensibility: Further types to be added in the future will have the basic fields plus extra fields
specific to their type.

### Domain synchronization

We enforce the same size for all `FieldVectors`, effectively leaving the extra SPH fields unused for dark matter particles.
This means that we can call `domain.sync` or `domain.syncGrav` as is and pass all fields in use by the propagator.

### Particle ordering

Currently, particles are ordered by their SFC key. To support multiple particle types, we support two orderings:
* Order-1: sorted by SFC key, same as existing ordering. Used for domain sync, halo exchange, gravity, and h computation.
* Order-2: sorted by type first, then by SFC key (global stable partition). Used for SPH computation.

### Order-2: Global stable partition by type

Order-2 is achieved by a **global stable partition** of all particles (assigned + halos) by type.
Since DM=0 < gas=1, a stable partition places all DM particles first (preserving their SFC order),
then all gas particles (preserving their SFC order).

After partition, the array looks like:
```
[all DM particles, sorted by SFC key] [all gas particles, sorted by SFC key]
```

The gas sub-array is itself a valid SFC-sorted particle set with its own halo structure. The gas
assigned range is found by comparing gas particle SFC keys with the domain's SFC key boundaries
(`domainKeyStart`, `domainKeyEnd`). This produces four indices:
  - `gasStart`: first gas particle (start of gas section = first gas halo)
  - `gasAssignedStart`: first assigned gas particle (key >= domainKeyStart)
  - `gasAssignedEnd`: last assigned gas particle (key < domainKeyEnd)
  - `gasEnd`: last gas particle (end of gas section = last gas halo)

The gas sub-array structure mirrors the full array:
```
Gas: [gasHaloFront | gasAssigned | gasHaloBack]
Full: [haloFront   | assigned    | haloBack   ]
```

### Gas-only layout and tree view

The octree structure (childOffsets, parents, internalToLeaf, centers, sizes) is reused unchanged —
it describes the spatial decomposition, independent of particle type.

A **gas-only layout** is computed by counting gas particles per leaf cell (scanning gas SFC keys
against cell boundaries). This layout maps the same tree cells to positions within the gas sub-array.

A gas-only `OctreeNsView` is created by copying the existing treeView and replacing the `layout`
pointer with the gas-only layout. The neighbor search (`findNeighbors`) uses `tree.layout` to
iterate particles per cell — with the gas layout, it only sees gas particles. Returned indices are
absolute positions in the full particle array, pointing to gas particle data.

### Gas-only halo exchange

During SPH computation, several intermediate halo exchanges are needed (for xm, vx/vy/vz/kx,
c11-c33/divv/c, prho/alpha, etc.). These use `domain.exchangeHalos()` which relies on
SendList/RecvList computed during sync in Order-1.

After partition to Order-2, these lists reference stale positions. Gas-only halo exchange lists must
be derived from:
  - The gas-only layout (which cells have gas particles and where)
  - The existing halo peer information (which ranks communicate)
  - The cell-to-rank assignment (which cells are assigned vs. halo)

This is a **local computation** (no MPI peer discovery needed) — remapping the same cell sets
through the gas layout produces gas-specific SendList/RecvList. These are stored as part of the
propagator's state and used for mid-SPH halo exchanges.

### Smoothing length: two-pass neighbor search

Smoothing length `h` is computed in two passes:

**Pass 1 (Order-1, all types):** `findNeighborsSph` with full layout and all particles. This
adjusts h for all particles (DM and gas) based on all-type neighbor counts. DM particles get their
final h values here (used for gravity softening). Gas particles get an initial h.

**Pass 2 (Order-2, gas only):** `findNeighborsSph` with gas layout and gas particles. This finds
gas-only neighbor lists for SPH computation and re-adjusts gas h based on gas-only neighbor counts.
Gas h from pass 1 is overwritten — this is accepted because:
  - Gas SPH kernels should use h tuned for gas-only neighbor density
  - VE formulation is robust to non-uniform neighbor distributions
  - The gas-tuned h propagates to subsequent timesteps, which is physically appropriate

### SPH type-awareness

Dark matter particles are excluded from all SPH interactions (density, pressure, momentum, energy).
In Order-2, the gas-only layout ensures `findNeighbors` only iterates gas particle ranges per cell.
SPH kernels receive gas-only neighbor lists and operate on the gas assigned range
`[gasAssignedStart, gasAssignedEnd)`. No per-interaction type filtering is needed.

### Acceleration accumulation

In the new propagator, gravity is computed before SPH. This requires changing SPH momentum kernels
from assignment to accumulation for accelerations:
  - `ax[i] = ...` → `ax[i] += ...` (same for ay, az)
  - `ax, ay, az` are zeroed before the gravity phase
  - `du[i] = ...` remains as assignment (no gravitational contribution to internal energy rate)

This matches what `NbodyProp` already does for zeroing accelerations.

### Initial conditions

Mixed particle type initial conditions are specified in a single file with a `type` column.
The type field is read during initialization alongside other particle fields.

## Splitting SimulationData and ParticlesData

### Current structure
`SimulationData<AccType>` holds:
  - `hydro`: `ParticlesData<AccType>` (all fields + all scalar parameters)
  - `chem`: `cooling::ChemistryData<RealType>`
  - `comm`: `MPI_Comm`

`ParticlesData` currently contains both basic and SPH fields, plus all scalar simulation parameters.

### New structure
`SimulationData<AccType>` becomes the composition container and the template parameter for propagators.
It holds:

**Basic particle fields** (moved from ParticlesData into SimulationData directly or into a BasicParticlesData member):
  - All basic FieldVectors: keys, x, y, z, x_m1, y_m1, z_m1, m, h, vx, vy, vz, ax, ay, az, rung, id, type
  - General simulation scalars:
    - `iteration`, `numParticlesGlobal`, `numParticlesGlobalPrev`
    - `ttot`, `minDt`, `minDt_m1`
    - `etot`, `ecin`, `eint`, `egrav`, `linmom`, `angmom`
    - `g`, `eps`, `etaAcc` (gravity parameters)
    - `ng0`, `ngmax` (neighbor parameters)
    - `maxDtIncrease`
    - `removeUnconvergedParticles`
    - `totalNeighbors`, `localNeighbors`, `maxHalos`
    - `stackUsedNc`, `stackUsedGravity`
  - Data structures: `neighbors`, `treeView`, `traversalStack`
  - I/O: `outputFieldIndices`, `outputFieldNames`
  - Memory: `allocGrowthRate_`

**SPH fields** (remain in a dedicated SphFields or HydroData member):
  - All SPH FieldVectors: rho, temp, u, du, du_m1, p, prho, tdpdTrho, c, cv, mue, mui,
    divv, curlv, c11-c33, alpha, xm, kx, gradh, dV11-dV33, nc, ugrav, wh, whd
  - SPH-specific scalars:
    - `gamma`, `polytropic_index`, `polytropic_const`, `eosChoice`, `muiConst`, `soundSpeedConst`
    - `Kcour`, `Krho` (SPH timestep fractions)
    - `minDtCourant`, `minDtRho` (SPH timestep constraints)
    - `alphamin`, `alphamax`, `decay_constant` (artificial viscosity parameters)
    - `Atmin`, `Atmax`, `ramp` (artificial viscosity limiter constants)
    - `sincIndex`, `kernelChoice`, `K` (SPH kernel parameters)

**Chemistry data**: `chem` (unchanged)

**MPI communicator**: `comm` (unchanged)

## Timestep cycle for HydroDarkProp

The new propagator computes gravity and SPH in separate phases with a global partition switch:

```
 1. domain.syncGrav()                — Order-1 (SFC), all particles
 2. fill halos                       — Order-1, basic + SPH fields
 3. findNeighborsSph pass 1          — Order-1, all types, adjusts h for all particles
 4. zero ax, ay, az                  — all particles
 5. gravity (all particles)          — Order-1, uses octree, ax/ay/az += gravity
 6. stable partition by type         — all particles (assigned+halos) → Order-2
 7. find gas assigned range          — SFC key comparison with domain boundaries
 8. build gas-only layout            — recount gas particles per cell, same tree structure
 9. build gas-only treeView          — copy treeView, swap layout pointer
10. build gas-only halo lists        — remap cell sets through gas layout (local, no MPI)
11. findNeighborsSph pass 2          — Order-2, gas only, gas-only neighbors, adjusts gas h
12. SPH kernels (gas only)           — xmass, VE, IAD, divv/curlv, EOS, AV, momentum+energy
                                       with gas-only halo exchanges between sub-steps
                                       ax/ay/az += SPH forces, du = SPH energy rate
13. reverse partition                — restore Order-1 from stored permutation
14. integrate (all particles)        — Order-1, update positions/velocities
15. → next iteration
```

Steps 6 and 13 are inverse operations using the same permutation vector. The partition is O(N)
(stable partition / gather+scatter with buffer). All arrays (basic + SPH) are permuted together.

### Block-timestep (BDT) interaction

For `HydroDarkBdtProp` (GPU-required block-timestep variant):
  - **Full sync** (substep 0): rebuild everything — partition, gas layout, gas halo lists
  - **Partial sync** (substeps 1+): only positions change, SFC keys unchanged. The partition
    is reused — no re-sort needed. Gas layout and halo lists remain valid.

## Implementation plan

### Phase 1: Add the type field
Add `type` as a `FieldVector<uint8_t>` to the basic fields. Add it to `fieldNames`, `dataTuple()`,
and `FieldVariant`. Set type = 1 (gas) for all particles in existing initializers to maintain backward
compatibility. Add type to I/O (read/write).

### Phase 2: Split ParticlesData
Refactor `ParticlesData` by moving basic fields and general scalars into `SimulationData` (or a
`BasicParticlesData` member). SPH-specific fields and scalars go into a dedicated `SphFields` member.
Update all propagators and callers to access fields through the new structure. `SimulationData`
becomes the propagator template parameter.

### Phase 3: Implement global type partition and gas-only layout
Implement the Order-1 ↔ Order-2 global partition:
  - Stable partition of all particles (assigned + halos) by type, preserving SFC order within type
  - Store the permutation vector for reverse operation
  - Find gas assigned range by SFC key binary search against domain boundaries
  - Build gas-only layout by counting gas particles per leaf cell
  - Build gas-only OctreeNsView by swapping layout pointer
  - Build gas-only halo exchange lists by remapping cell sets through gas layout

### Phase 4: Implement HydroDarkProp
Build the new propagator following the timestep cycle above:
  - Two-pass neighbor search: Order-1 for h (all types), Order-2 for SPH (gas only)
  - Gravity-first with acceleration zeroing and accumulation (ax/ay/az += only, du stays =)
  - Global partition around SPH phase with gas-only layout and gas-only halo exchange
  - Register in propagator factory as "hydro-dark"
  - BDT variant: full sync rebuilds partition, partial sync reuses it

### Phase 5: Mixed initial conditions
Extend file-based initialization to read a `type` column from HDF5/ASCII input.
Add or adapt an initializer that sets up mixed DM + gas particle distributions.

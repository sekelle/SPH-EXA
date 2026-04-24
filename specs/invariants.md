# Particle Species — Invariants

Invariants that must hold at all times during simulation with multiple particle types.
Each invariant is testable and maps to one or more GTest assertions.

## Conservation invariants

### INV-1: Total particle count preserved
The global particle count (`numParticlesGlobal`) must remain constant across domain sync,
reindexing, and timestep integration. No particles are created or destroyed during normal
time evolution.

### INV-2: Particle type immutable during evolution
The `type` field of each particle must not change during domain sync, reindexing, halo
exchange, or time integration. A particle's type is set at initialization and remains fixed.

### INV-3: Mass conservation
The sum of all particle masses across all ranks must remain constant (within floating-point
tolerance) across timesteps, domain syncs, and reindexing operations.

### INV-4: Particle identity preserved across reindexing
After any Order-1 ↔ Order-2 reindex and its inverse, each particle's `id` must map to the
same field values it had before the operation (positions, velocities, mass, type, all fields).

## Ordering invariants

### INV-5: Order-1 is pure SFC
In Order-1, all assigned particles in `[startIndex, endIndex)` are sorted by SFC key.
This is the existing invariant, unchanged.

### INV-6: Order-2 is type-then-SFC globally
In Order-2, all DM particles (type=0) appear before all gas particles (type=1) in the
array. Within each type group, particles are sorted by SFC key. The gas sub-array forms
a contiguous SFC-sorted set with its own halo structure.

### INV-7: Partition round-trip identity
Applying the Order-1 → Order-2 partition followed by its inverse produces the original
particle ordering. No data is lost or corrupted.

### INV-8: Layout consistency
In Order-1, `layout[i+1] - layout[i]` equals the number of particles in leaf cell `i`.
The gas-only layout maps the same tree cells to gas particle ranges. The sum of gas
particles across all cells equals the total gas particle count (assigned + halos).

## Type-physics invariants

### INV-9: DM excluded from SPH
Dark matter particles (type=0) must never appear in SPH neighbor lists, SPH kernel
computations, or have their SPH fields (rho, p, u, etc.) modified by SPH kernels.

### INV-10: DM participates in gravity
Dark matter particles must participate in gravitational force computation. Their
accelerations (ax, ay, az) must include gravitational contributions.

### INV-11: Gas participates in both forces
Gas particles (type=1) must have accelerations that include both gravitational and SPH
contributions after the force computation phase.

### INV-12: Smoothing length two-pass
DM smoothing length `h` is computed from all-type neighbors (pass 1, Order-1).
Gas smoothing length `h` is initially set from all-type neighbors (pass 1), then
re-adjusted from gas-only neighbors (pass 2, Order-2). The gas-tuned h is the
final value used for SPH computation.

## Field invariants

### INV-13: Uniform array sizes
All `FieldVector` arrays (basic + SPH) must have identical sizes at all times, equal to
`bufDesc.size` (assigned + halos). SPH field entries for DM particles are allocated but
unused.

### INV-14: SPH fields unchanged for DM
SPH field values at indices corresponding to DM particles must not be modified by any
SPH kernel. They may contain arbitrary/uninitialized values.

### INV-15: Acceleration accumulation
After the combined gravity + SPH force computation, `ax[i]` for gas particles must equal
the sum of gravitational and SPH accelerations. For DM particles, `ax[i]` must equal the
gravitational acceleration only.

## Domain sync invariants

### INV-16: Type field survives domain sync
The `type` field must be included in domain sync particle exchange and halo exchange.
After sync, each particle retains its correct type value.

### INV-17: Halo particles have correct types
Halo particles received from other ranks must have their `type` field correctly
transferred. The type of a halo particle must match the type of the corresponding
assigned particle on the sending rank.

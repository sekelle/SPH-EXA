# Feature: Mixed Initial Conditions (Phase 5)

Extend file-based initialization to read a `type` column, and support mixed DM + gas
particle distributions.

## Scenarios

### Scenario: HDF5 file with type column loaded correctly
Given: an HDF5 file with datasets x, y, z, vx, vy, vz, m, h, type
And: type values are a mix of 0 (DM) and 1 (gas)
When: the file is loaded via `--init file:<path>`
Then: all particles have correct positions, velocities, masses, and types
Test: `TEST_F(MixedInit, hdf5WithTypeColumn)`

### Scenario: HDF5 file without type column defaults to gas
Given: an HDF5 file without a type dataset (legacy format)
When: the file is loaded
Then: all particles have type = 1 (gas) — backward compatible
Test: `TEST_F(MixedInit, hdf5WithoutTypeDefaultsGas)`

### Scenario: ASCII file with type column loaded correctly
Given: an ASCII file with columns x, y, z, vx, vy, vz, m, h, type
When: the file is loaded
Then: particle types are correctly parsed
Test: `TEST_F(MixedInit, asciiWithTypeColumn)`

### Scenario: Type counts reported at startup
Given: a mixed DM + gas initialization
When: the simulation starts
Then: the number of DM and gas particles is reported per rank and globally
Test: `TEST_F(MixedInit, typeCountsReported)`

### Scenario: Built-in initializers set type = gas
Given: any built-in initializer (--init sedov, noh, evrard, etc.)
When: particles are created
Then: all particles have type = 1 (gas)
And: the simulation behaves identically to the pre-species version
Test: `TEST_F(MixedInit, builtInInitSetGas)`

### Scenario: Type field written to output
Given: a mixed simulation running with file output enabled
When: output is written (HDF5 or ASCII)
Then: the type field is included in the output file
And: values match the current particle types
Test: `TEST_F(MixedInitIO, typeFieldInOutput)`

### Scenario: Checkpoint restart preserves types
Given: a mixed simulation that writes a checkpoint
When: the simulation is restarted from the checkpoint
Then: all particle types are correctly restored
And: the simulation continues identically
Test: `TEST_F(MixedInitIO, checkpointRestoreTypes)`

### Scenario: Mixed init with Evrard collapse + DM halo
Given: an initializer that creates a gas cloud (type=1) surrounded by a DM halo (type=0)
When: the simulation runs for 10 timesteps
Then: DM particles orbit gravitationally
And: gas particles undergo hydrodynamic collapse + gravitational interaction
Test: `TEST_F(MixedInit, evrardWithDmHalo)` (integration)

## Invariants tested
- INV-1 (particle count preserved after init + sync)
- INV-2 (type values unchanged after init)
- INV-13 (uniform array sizes with both types present)

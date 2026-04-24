# Particle Species — Specifications

Behavioral specifications for the particle species feature (branch `particle-species`).
Each feature spec maps directly to GTest test cases.

## Structure

```
specs/
├── invariants.md              — System invariants that must always hold
├── features/
│   ├── type-field.md          — Phase 1: type field addition
│   ├── simulation-data.md     — Phase 2: SimulationData/ParticlesData split
│   ├── type-reindexing.md     — Phase 3: Order-1 ↔ Order-2 permutation
│   ├── hydro-dark-prop.md     — Phase 4: HydroDarkProp propagator
│   └── mixed-init.md          — Phase 5: Mixed initial conditions
├── architecture/
│   └── test-map.md            — Spec → GTest mapping
└── README.md                  — This file
```

## Testing methodology

Specs are written as behavioral scenarios following the pattern:

```
### Scenario: <descriptive name>
Given: <preconditions>
When: <action>
Then: <expected outcome>
Test: <GTest suite and test name>
```

Each scenario maps to exactly one `TEST()` or `TEST_F()` in the codebase.
Test names follow the convention `TEST(FeatureSuite, scenarioName)` where the
scenario name is a readable description matching the spec.

Tests are organized into:
- **Unit tests**: per-module, single-rank, fast
- **MPI tests**: multi-rank domain sync and halo exchange
- **GPU tests**: CUDA/HIP parity with CPU
- **Integration tests**: full timestep cycle with mixed particle types

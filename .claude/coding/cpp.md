# SPH-EXA — C++ Coding Standards

Extends `.claude/guidelines/cpp.md` with project-specific conventions.

## Module Structure

7 modules under project root:
- `domain/` — Cornerstone octree library (namespace: `cstone`)
- `sph/` — SPH physics kernels (namespace: `sph`, `sph::gpu`)
- `ryoanji/` — N-body gravity solver (namespace: `ryoanji`)
- `main/` — Application frontend (namespace: `sphexa`)
- `physics/` — Additional physics modules
- `scripts/` — Python utility scripts
- `cmake/` — CMake modules

## Namespaces

- `cstone` — domain decomposition, octree, SFC keys, halos
- `sph` — SPH physics implementations
- `sph::gpu` — GPU-accelerated SPH kernels
- `sphexa` — application-level code (propagators, init, I/O)
- `ryoanji` — gravity solver

No `using namespace` in headers. Acceptable in `.cpp`/`.cu` implementation files.

## Header-Only Libraries

SPH kernels and most domain utilities are header-only. Keep implementation
in headers when templates are involved. Use `.hpp` extension.

## ParticlesData Pattern

`ParticlesData` is the central particle data container. It uses:
- Template parameter for accelerator type (`CpuTag`/`GpuTag`)
- `FieldVector` for dynamically-sized particle arrays
- Field registration via variant types
- Resize operations must preserve existing data

Current branch (`particle-species`) is splitting this into:
- Basic fields class (all particle types): positions, velocities, mass, keys, type
- SPH fields class (gas particles only): density, pressure, energy, etc.

## Propagator Pattern

Propagators drive time-stepping. They:
- Inherit from `IPropagator` interface
- Use template parameters to select physics models
- Register which fields they need
- Call domain sync, SPH kernels, gravity in sequence

Existing: `HydroVeProp`, `HydroStdProp`, `HydroDiskProp`, `HydroGrackleProp`
Planned: `HydroDarkProp` (gas + dark matter)

## Accelerator Abstraction

CPU/GPU polymorphism via:
```cpp
template<class AccType>
void compute(...)
{
    if constexpr (cstone::HaveGpu<AccType>{})
    {
        // GPU kernel launch
    }
    else
    {
        // CPU implementation with OpenMP
    }
}
```

## Domain Synchronization

- `domain.sync()` — exchange particles between MPI ranks based on SFC ordering
- `domain.syncGrav()` — sync with gravity-specific halo exchange
- All `FieldVector`s must be same size (even unused fields for some particle types)
- Halo exchange via octree-based discovery

## Field System

- Fields are `std::vector<T>` wrapped in `FieldVector`
- Dynamic field activation per propagator
- Field names used for I/O selection (`-f x,y,z,rho,p`)
- Variant-based field registry for type-erased access

## MPI Patterns

- All ranks participate in domain decomposition
- Collective operations for global reductions (min dt, total energy)
- Point-to-point for halo exchange
- No `MPI_COMM_WORLD` hardcoding — pass communicators explicitly

## OpenMP Patterns

- `#pragma omp parallel for` on particle loops
- Schedule clause based on workload (static for uniform, dynamic for irregular)
- Reduction clauses for accumulations
- No nested parallelism

## Testing Conventions

- GTest `TEST()` for standalone tests
- GTest `TEST_F()` for fixture-based tests
- Test files: `test_<component>.cpp`
- MPI tests: use `MPI_Init`/`MPI_Finalize` in test main
- GPU tests: separate `unit_cuda/` directories
- Analytical solution comparison tests in `main/src/analytical_solutions/`

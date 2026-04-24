# SPH-EXA

High-performance Smoothed Particle Hydrodynamics code for astrophysical simulations,
targeting exascale supercomputers. C++20, CUDA/HIP, MPI, OpenMP.

## Quick reference

- **Build**: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON && cmake --build build -j$(nproc)`
- **Test**: `ctest --test-dir build --output-on-failure`
- **Run**: `OMP_NUM_THREADS=4 ./build/main/src/sphexa/sphexa --init sedov -n 50 -s 100 -w 10`
- **MPI**: `mpirun -np 2 ./build/main/src/sphexa/sphexa --init sedov -n 50 -s 100`
- **Format**: `clang-format -i <file>` (config in `.clang-format`, 120-col, Allman, 4-space)

## Modules

| Module | Path | Namespace | Purpose |
|--------|------|-----------|---------|
| Cornerstone | `domain/` | `cstone` | Octree, SFC keys, domain decomposition, halos, neighbor search |
| SPH | `sph/` | `sph` | Physics kernels: hydro_ve, hydro_std, hydro_turb, EOS |
| Ryoanji | `ryoanji/` | `ryoanji` | N-body gravity: fast multipole, GPU tree traversal, Ewald |
| Main | `main/` | `sphexa` | Propagators, init conditions, I/O, CLI |
| Physics | `physics/` | — | Disk physics, GRACKLE cooling |
| Scripts | `scripts/` | — | Python post-processing utilities |

## Key types

- `ParticlesData<AccType>` — particle field container (`sph/include/sph/particles_data.hpp`)
- `SimulationData<AccType>` — top-level data holder (`main/src/sphexa/simulation_data.hpp`)
- `Propagator<DomainType, DataType>` — abstract timestep driver (`main/src/propagator/ipropagator.hpp`)
- `Domain<KeyType, T, Acc>` — distributed octree domain (`domain/include/cstone/domain/domain.hpp`)
- `AccType` = `CpuTag` or `GpuTag` — compile-time CPU/GPU selection

## Propagators (--prop flag)

`ve` (default), `ve-bdt` (block timestep, GPU), `std`, `std-cooling` (GRACKLE),
`nbody` (gravity only), `turbulence`, `turbulence-ve`, `std-disk`

## Test structure

Google Test. Tests in `<module>/test/`. MPI tests in `integration_mpi/`. GPU tests in `unit_cuda/`.

## Current branch: particle-species

Adding multiple particle types (DM + gas). See `design.md` for the full design and
`specs/` for feature specifications, invariants, architecture, and implementation plan.

## Workflow instructions

See `.claude/CLAUDE.md` for role definitions, workflow routing, coding guidelines,
and project commands.

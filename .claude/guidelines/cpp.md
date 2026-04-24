# C++20 Guidelines

## Version & Tooling

- C++20 standard (C++17 for CUDA kernel code)
- Compilers: GCC 12+, Clang 16+
- Format: `clang-format` with project `.clang-format` (120-char column, Allman braces, 4-space indent)
- Lint: `clang-tidy` with project `.clang-tidy`
- Build system: CMake 3.24+
- Test framework: Google Test (GTest), auto-fetched via CMake

## Style

- Follow project `.clang-format` configuration
- 120-character column limit
- 4-space indentation, no tabs
- Allman brace style
- Align consecutive assignments and trailing comments
- Template declarations always break before body

## Naming Conventions

- **Namespaces**: lowercase (e.g., `cstone`, `sph`, `sphexa`, `ryoanji`)
- **Classes/Structs**: CamelCase (e.g., `ParticlesData`, `Domain`, `HydroVeProp`)
- **Member variables**: camelCase (e.g., `numParticlesGlobal`, `minDt`)
- **Functions**: camelCase (e.g., `computeVeImpl`, `syncHalos`, `exchangeHalos`)
- **Type aliases**: CamelCase (e.g., `RealType`, `KeyType`, `AcceleratorType`)
- **Template parameters**: single uppercase or descriptive CamelCase (e.g., `T`, `Dataset`, `AccType`)
- **Constants/Macros**: UPPER_CASE
- **GPU namespaces**: nested `gpu::` (e.g., `sph::gpu::`)

## Includes

- Project headers first, then external, then stdlib
- Use `#pragma once` for header guards
- Forward declare when possible to minimize include chains

## Templates

- Extensive use of C++20 templates and concepts
- `if constexpr` for compile-time GPU/CPU branching (e.g., `cstone::HaveGpu<AccType>{}`)
- Policy-based design via template parameters for physics models
- SFINAE and concepts for constraining template interfaces

## Memory & Performance

- Prefer stack allocation and value semantics
- Use `std::vector` for dynamic particle data
- Minimize allocations in hot loops
- Profile before optimizing; measure with performance tests

## Error Handling

- Use return codes or exceptions consistently within a module
- MPI errors: check return values, use `MPI_Abort` for unrecoverable failures
- File I/O: validate paths and handle missing files gracefully

## Safety

- No raw `new`/`delete` in new code; use RAII and smart pointers
- Bounds checking in debug builds
- Thread safety via OpenMP pragmas, not manual locking

# CI/CD Guidelines

## Pipeline Structure (three-stage)

Every change follows: **Build → Validate → Test**

### Build Stage

- CMake configure + build (Debug and Release)
- Build with and without GPU support
- Build with and without optional dependencies (HDF5, GRACKLE)

### Validate Stage

- Formatting check: `clang-format --dry-run --Werror`
- Static analysis: `clang-tidy`
- CMake configuration validation

### Test Stage

- Unit tests via CTest
- MPI integration tests (multi-rank)
- GPU tests (when hardware available)
- Performance regression tests (optional)

## Triggers

- Push to `develop` or `master`
- PRs against `develop` or `master`
- Path exclusions: `docs/**`, `*.md`, `LICENSE`, `scripts/`

## Caching

- CMake build directory caching
- Dependency download caching (GTest, H5hut)

## Testing Matrix

- Compilers: GCC 12+, Clang 16+
- GPU: CUDA 12+, HIP/ROCm 6+ (where available)
- MPI implementations: OpenMPI, MPICH
- Configurations: CPU-only, CUDA, HIP

# CMake Build System Guidelines

## Version & Structure

- CMake 3.24+ required
- Root `CMakeLists.txt` configures project-wide options
- Each module has its own `CMakeLists.txt` via `add_subdirectory()`
- Module path: `cmake/` for custom modules, `main/cmake/` for additional

## Build Configuration

- Default build type: Release (`-O3 -march=native -DNDEBUG`)
- Debug builds for development and testing
- Options:
  - `BUILD_TESTING` (ON) — unit and integration tests
  - `BUILD_ANALYTICAL` (ON) — analytical solution comparisons
  - `SPH_EXA_WITH_CUDA` (ON) — NVIDIA GPU support
  - `SPH_EXA_WITH_HIP` (ON) — AMD GPU support
  - `SPH_EXA_WITH_H5HUT` (ON) — HDF5 I/O
  - `SPH_EXA_WITH_GRACKLE` (OFF) — radiative cooling
  - `SPH_EXA_WITH_DISKS` (OFF) — disk physics
  - `INSITU` (None|Catalyst|Ascent) — in-situ visualization

## Build Commands

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON

# Build
cmake --build build -j$(nproc)

# Test
ctest --test-dir build --output-on-failure

# Build specific target
cmake --build build --target <test_name>
```

## Dependencies

- **Required**: MPI, OpenMP
- **Optional**: CUDA Toolkit, HIP/ROCm, HDF5, H5hut, ParaView Catalyst, Ascent, GRACKLE
- **Auto-fetched**: Google Test (via `setup_GTest` CMake module)
- Use `FetchContent` or custom `Fetch_*.cmake` for downloaded dependencies

## Testing with CTest

- Tests registered via `add_test()` in module CMakeLists.txt
- MPI tests use `mpirun -np <N>` as test command prefix
- GPU tests only run when GPU hardware detected
- Use `ctest -R <pattern>` to run specific test subsets

## Conventions

- Use modern CMake targets (`target_link_libraries`, `target_include_directories`)
- Prefer `PRIVATE` linkage unless headers are part of public API
- Use generator expressions for conditional settings
- Keep `CMakeLists.txt` files readable and well-commented

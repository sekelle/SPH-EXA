# CUDA/HIP GPU Programming Guidelines

## Version & Tooling

- CUDA Toolkit 12.0+ with C++20 support
- HIP/ROCm 6.0+ with C++20 support
- CUDA and HIP are mutually exclusive (`FATAL_ERROR` if both enabled)
- GPU-aware MPI optional (`-DCSTONE_WITH_GPU_AWARE_MPI=ON`)

## File Conventions

- `.cu` files for CUDA kernel implementations
- `.cuh` files for CUDA device headers
- Host-callable GPU functions declared in `.hpp` headers
- GPU implementations in `gpu::` sub-namespace

## Kernel Patterns

- Warp-aware algorithms (especially in ryoanji tree traversal)
- Use shared memory for neighbor lists and reduction operations
- Prefer thrust/CUB primitives for sort, scan, reduce
- For HIP: use rocThrust and hipCUB equivalents

## Memory Management

- Device memory managed through RAII wrappers where possible
- Minimize host-device transfers; batch transfers when needed
- Use pinned memory for async transfers
- Pre-allocate buffers; avoid per-timestep allocations

## CPU/GPU Abstraction

- `cstone::CpuTag` / `cstone::GpuTag` accelerator type tags
- `AccType` template parameter selects CPU or GPU code paths
- `if constexpr (cstone::HaveGpu<AccType>{})` for compile-time branching
- Shared headers with `__host__ __device__` annotations where needed

## Testing

- GPU unit tests in `<module>/test/unit_cuda/`
- Test both CPU and GPU code paths
- Use small problem sizes for unit tests (fast execution)
- Performance tests separate from correctness tests

## Build System

- CMake `check_language(CUDA)` / `check_language(HIP)` for detection
- `CMAKE_CUDA_STANDARD 20` / `CMAKE_HIP_STANDARD 20`
- Separate compile flags for CUDA release builds (`-O3 -DNDEBUG`)

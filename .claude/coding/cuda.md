# SPH-EXA — CUDA/HIP Coding Standards

Extends `.claude/guidelines/cuda.md` with project-specific conventions.

## GPU Kernel Organization

### Domain (Cornerstone)
- `domain/include/cstone/primitives/` — GPU primitives (sort, scan, gather)
- `domain/include/cstone/tree/` — GPU octree construction
- `domain/include/cstone/halos/` — GPU halo exchange support

### SPH
- `sph/include/sph/*.cu` — GPU SPH kernel implementations
- Mirror CPU implementations in `.hpp` with GPU variants in `.cu`
- GPU kernels in `sph::gpu` namespace

### Ryoanji (Gravity)
- `ryoanji/src/ryoanji/nbody/traversal_gpu.cu` — warp-aware tree traversal
- `ryoanji/src/ryoanji/nbody/direct.cuh` — direct N-body on GPU
- `ryoanji/src/ryoanji/nbody/upsweep_gpu.cu` — multipole upsweep

## Warp-Aware Patterns (Ryoanji)

The gravity solver uses warp-level primitives for tree traversal:
- One warp processes one target particle's interaction list
- `__shfl_sync` for warp-level communication
- Shared memory for interaction list staging
- Critical for performance — do not break warp-coherent access patterns

## Thrust/CUB Usage

- Use thrust for high-level operations (sort, reduce, scan)
- Use CUB for block/warp-level primitives
- For HIP: rocThrust and hipCUB are drop-in replacements
- Avoid raw CUDA API when thrust/CUB provides the operation

## Memory Patterns

- Particle arrays allocated once, resized only during domain sync
- Device-side neighbor lists in shared memory during kernel execution
- Avoid per-particle dynamic allocation on device
- Use `cudaMemcpyAsync` with streams for overlapping compute and transfer

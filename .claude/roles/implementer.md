# Role: Implementer

Implement ONE feature at a time, strictly within architectural constraints.
Build against the design, not around it.

## Orient before coding (every session)

Read: design.md, module structure, existing tests for YOUR modules,
data structures in the relevant headers.

Summarize: "I am implementing [feature]. Modules: [X]. Dependencies: [Y].
Tests: [N existing]. Current state: [description]."

## Boundary discipline

**Must NOT**: modify module interfaces without discussion, access another
module's internal data structures, add undeclared dependencies.

**Must**: implement all specified functionality, conform to existing data
structures, maintain existing test coverage, handle failure modes.

## Implementation protocol (TDD)

1. Pick a specific behavior to implement
2. Write test for that behavior
3. Run — should fail (red)
4. Implement minimum to pass (green)
5. Run ALL previous tests — must still pass
6. Refactor if needed, re-run everything
7. Next behavior

One behavior at a time. No batching.

## Constraints

### C++20 (all simulation code)
- Latest stable C++20 features
- Templates for CPU/GPU abstraction via AccType
- Error handling: return codes or exceptions consistently per module
- No raw new/delete; RAII and value semantics
- Follow existing clang-format configuration

### GPU (CUDA/HIP)
- Mirror CPU implementations in GPU kernels
- GPU code in `gpu::` sub-namespaces
- Test both CPU and GPU paths
- Use thrust/CUB for standard operations

### MPI
- Pass communicators explicitly (no MPI_COMM_WORLD hardcoding)
- Check return values on MPI calls
- Test with multiple ranks

## Module-specific notes

### Domain (Cornerstone octree)
- SFC key ordering (Morton/Hilbert)
- Domain decomposition across MPI ranks
- Halo discovery and exchange
- Neighbor finding via octree traversal

### SPH Physics
- Multiple formulations: standard, volume-element (VE), turbulent
- Kernel functions (cubic spline, Wendland)
- Equation of state implementations
- Field-based particle data access

### Ryoanji (Gravity)
- Fast multipole method with warp-aware GPU traversal
- Cartesian multipole expansions
- Ewald summation for periodic boundaries
- Direct summation for close interactions

### Main (Application)
- Propagators: time-stepping drivers selecting physics
- Initial conditions: Sedov, Noh, Evrard, KH, turbulence, etc.
- I/O: HDF5 via H5hut, ASCII fallback
- CLI argument parsing

## When stuck

Document the issue:
```
Type: Design Gap | Interface Conflict | Test Ambiguity
Module: [which]
What I need: [specific]
What's blocking: [which artifact]
Proposed resolution: [if any]
Impact: [can I continue with other work?]
```

## Code quality

- Naming follows existing conventions (camelCase functions, CamelCase types)
- Explicit typed errors, no silent failures
- No implicit state. State visible through function signatures.
- Non-obvious paths get WHY comments referencing design requirements.

## Definition of Done (per feature)

- [ ] All specified behaviors have corresponding tests
- [ ] Tests pass on CPU path
- [ ] Tests pass on GPU path (if applicable)
- [ ] MPI tests pass with multiple ranks (if applicable)
- [ ] No regressions in existing test suite
- [ ] clang-format clean
- [ ] No undeclared dependencies
- [ ] Design doc alignment verified
- [ ] No TODO comments without linked issue
- [ ] Error paths tested (not just happy path)

## Session management

End: tests passing/total, issues filed, remaining work planned,
full test suite results. Last session: run full suite, report regressions,
declare complete only if all DoD items checked.

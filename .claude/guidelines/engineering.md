# General Engineering Guidelines

## Commits & Branching

- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`, `ci:`
- Branch naming: `feature/`, `fix/`, `docs/`, `refactor/`, `test/`
- One logical change per commit; reference issue numbers where applicable

## Error Handling

- Never swallow errors silently
- Use exceptions or return codes consistently within a module
- Validate at system boundaries (user input, file I/O, MPI communication)
- Trust internal code paths within a module

## Code Organization

- Includes grouped: project headers → external → stdlib (blank line between groups)
- Public interface before private implementation in header files
- One component/responsibility per file; keep files under 500 lines where practical
- No globals; pass dependencies explicitly through function parameters or class members

## Code Quality

- clang-format enforces formatting (`.clang-format` in repo root)
- clang-tidy checks for common issues (`.clang-tidy` in repo root)
- No hardcoded secrets, tokens, or credentials in source
- Keep dependencies updated; use CMake FetchContent for third-party libraries

## Testing Philosophy

### Test-Driven Development

- Write a failing test before writing implementation code where practical
- Tests describe behavior, not implementation details
- Test names should read as specifications: `TEST(ParticlesData, resizePreservesExistingData)`

### Test Organization

- Unit tests: co-located in `<module>/test/` directories
- Integration tests: MPI-enabled tests in `<module>/test/integration_mpi/`
- Performance tests: `<module>/test/performance/`
- GPU tests: `<module>/test/unit_cuda/` for CUDA-specific tests
- All tests use Google Test (GTest) framework

### Test Patterns

- Use `TEST()` and `TEST_F()` with descriptive names
- Arrange-Act-Assert structure
- Test edge cases and error paths, not just happy paths
- Use `EXPECT_*` for non-fatal checks, `ASSERT_*` when continuation is meaningless
- MPI tests: verify behavior across multiple ranks

## Architecture Decision Records

- Design decisions documented in `design.md` or dedicated docs
- Record the context, decision, and consequences
- Update when design evolves

## Workflow Phases

1. **Analyst** — domain model, physics requirements, behavioral specs
2. **Architect** — module boundaries, data structures, interfaces
3. **Adversary** — challenge completeness, find blind spots, failure modes
4. **Implementer** — code against design, TDD per component
5. **Auditor** — test depth, confidence levels, coverage gaps
6. **Integrator** — cross-module integration, MPI correctness, GPU/CPU parity

# Workflow Router

Role definitions in `.claude/roles/`. Read the relevant role file when
activating a mode. These are behavioral constraints, not suggestions.

## Standards

Engineering guidelines in `.claude/guidelines/` (general, cross-project):
- `engineering.md` — commits, errors, code org, testing philosophy
- `cpp.md` — C++20 tooling, style, clang-format, clang-tidy
- `cuda.md` — CUDA/HIP GPU programming conventions
- `cmake.md` — CMake build system, CTest, dependencies
- `ci.md` — CI/CD pipeline structure
- `docs.md` — documentation requirements

Project-specific coding standards in `.claude/coding/`:
- `cpp.md` — SPH-EXA C++: template patterns, namespace policy, GPU abstractions
- `cuda.md` — SPH-EXA CUDA/HIP: kernel conventions, memory management
- `python.md` — SPH-EXA Python: utility scripts for post-processing

## Pre-commit discipline

Before committing: run `cmake --build build --target all` and `ctest --test-dir build`.
Use `/project:verify` for the full checklist.

## Automatic command invocation

| Command | When to invoke automatically |
|---|---|
| `/project:status` | **First message of every new session.** Establishes project state before any work. |
| `/project:verify` | **Before every commit.** Do not commit without running this. If it fails, fix and re-run. |
| `/project:spec-check` | **After completing a module change.** Validates design doc alignment before moving on. |

## Mode detection (every response)

### Step 1: Project state

1. `design.md` exists? -> Design document present
2. Source code with tests exists? -> Brownfield with baseline
3. Branch context? -> Check current branch for feature context

### Step 2: User intent -> mode -> role

| Intent | Mode | Role |
|--------|------|------|
| status | ASSESS | Read project state |
| audit [X] | AUDIT | auditor |
| implement / add | FEATURE | Feature Protocol |
| fix / bug / error | BUGFIX | Bugfix Protocol |
| design / spec | DESIGN | Design Protocol |
| review / find flaws | REVIEW | adversary |
| integrate | INTEGRATE | integrator |
| continue / next | RESUME | Read last state |
| Unclear | ASK | |

### Step 3: Before acting, one line

```
Mode: [MODE]. Project: [state]. Role: [role]. Reason: [why].
```

## Role switching

On switch: `Switching to [role]. Previous: [role].`
Read `.claude/roles/[role].md`. Apply its constraints.

## Protocols

**Feature**: analyst -> design | architect -> interfaces | adversary -> gate 1 | implementer -> tests+code | auditor -> gate 2 | adversary -> findings | integrator (if cross-module). Done = tests pass + design aligned.

**Bugfix**: diagnose -> failing test first -> fix -> audit depth -> update.

**Design**: new feature -> analyst | arch change -> architect | design doc update.

## Entry point

**Particle species implementation** (current branch): `design.md` describes adding particle types (dark matter, gas) to SPH-EXA. Key changes: splitting ParticlesData, new type field, new propagator `HydroDarkProp`, dual particle ordering (SFC key, then type+key). Enter via FEATURE mode with implementer role.

## Modules

- **domain/** — Cornerstone octree library (SFC keys, domain decomposition, halos, neighbor search)
- **sph/** — SPH physics kernels (hydro_ve, hydro_std, hydro_turb, kernels, EOS)
- **ryoanji/** — N-body gravity solver (multipole, GPU tree traversal)
- **main/** — Application frontend (propagators, init conditions, I/O, CLI)
- **physics/** — Additional physics (disk, cooling/GRACKLE)
- **scripts/** — Python utility scripts

## Escalation paths

Implementer -> Architect (interface/boundary) or Analyst (design gap).
Adversary -> Architect (structural) or Analyst (missing spec).
Auditor -> Implementer (weak tests) or Architect (contract divergence).
Integrator -> Architect (cross-module issues).

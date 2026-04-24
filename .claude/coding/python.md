# SPH-EXA — Python Coding Standards

Extends `.claude/guidelines/` general principles.

## Scope

Python in SPH-EXA serves utility purposes only:
1. **Post-processing scripts** — HDF5 data manipulation, visualization
2. **Initial condition generators** — creating particle setups
3. **Analysis scripts** — power spectrum, slice plots

Python is NOT used for any simulation code or performance-critical paths.

## Scripts

Located in `scripts/`:
- `add_m1.py` — Add previous timestep fields to HDF5 files
- `init_file.py` — Create initial conditions
- `plot_power.py` — Power spectrum plotting
- `slice.py` — Slice visualization

## Conventions

- Python 3.9+ for compatibility with HPC systems
- Use `numpy` and `h5py` for data processing
- Use `matplotlib` for plotting
- Scripts should be runnable standalone with `argparse` CLI
- No complex dependency chains — keep scripts self-contained

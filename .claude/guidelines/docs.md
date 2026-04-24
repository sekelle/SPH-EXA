# Documentation Maintenance

## Required Documentation

1. **README.md** — purpose, quick-start, compilation, test cases, usage examples
2. **design.md** — design documents for new features (e.g., particle types)
3. **CLAUDE.md** — project context for AI assistants
4. **domain/README.md** — Cornerstone octree library documentation

## Inline Documentation

- Doc comments explain WHY, not WHAT
- Doxygen-style comments: `@file`, `@brief`, `@author`, `@param`, `@return`
- Code comments only where logic isn't self-evident
- Module-level docs in header file preambles

## Keeping Docs Current

- README updated as part of any PR that changes build, test, or usage
- design.md updated when design evolves
- Stale docs are worse than no docs — delete rather than leave misleading

## Sphinx Documentation

- Source in `docs/source/` for ReadTheDocs
- Update when public API changes

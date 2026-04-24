Pre-commit verification. Run this before every commit claim.

1. Format: `find . -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' | head -20 | xargs clang-format --dry-run --Werror 2>&1` — check formatting
2. Build (Debug): `cmake --build build -j$(nproc) 2>&1` — must succeed with no errors
3. Unit tests: `ctest --test-dir build --output-on-failure 2>&1` — all must pass
4. Check for new files: `git status --short` — any untracked files that should be committed?
5. Check design alignment: verify changes match design.md intent

If ANY step fails, do NOT commit. Fix first, then re-run /project:verify.

Report: show pass/fail for each step.

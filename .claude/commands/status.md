Project status assessment. Run at the start of any session.

1. Check branch: `git branch --show-current` — which feature branch?
2. Check design docs: `cat design.md 2>/dev/null` — what's the current design?
3. Check source structure: `ls domain/include/ sph/include/ ryoanji/src/ main/src/ 2>/dev/null`
4. Check build: `cmake --build build --target all 2>&1 | tail -10` — does it compile?
5. Check tests: `ctest --test-dir build --output-on-failure 2>&1 | tail -20` — do tests pass?
6. Check git status: uncommitted changes? Ahead/behind remote?

Report:
- Current branch and its purpose
- Build status (compiles? warnings?)
- Test status (pass/fail counts)
- Recent changes (last 5 commits)
- Open issues or TODOs
- Recommended next action

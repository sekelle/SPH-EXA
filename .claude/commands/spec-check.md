Design consistency check. Validates that design docs and code stay aligned.

1. **Design doc compliance**: read `design.md`. For each specified behavior or
   data structure, check if corresponding code exists. Report: IMPLEMENTED /
   PARTIAL / MISSING per item.

2. **Naming consistency**: grep source files for type and function names
   mentioned in design docs. Flag names that don't match.

3. **Test coverage for design items**: for each design requirement, check if
   a test exists that verifies it. Report: COVERED / PARTIAL / NONE.

4. **Module boundary respect**: check that changes stay within the intended
   module boundaries per design. Flag cross-module changes that may need
   discussion.

Report summary table with pass/fail per check category.

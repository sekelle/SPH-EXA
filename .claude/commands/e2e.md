Run end-to-end simulation tests. Use after implementation is functional.

Steps:

1. Build release binary:
   - `cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON`
   - `cmake --build build-release -j$(nproc)`

2. Run standard test cases (single rank):
   - `./build-release/main/src/sphexa/sphexa --init sedov -n 20 -s 10 -w 5`
   - `./build-release/main/src/sphexa/sphexa --init noh -n 20 -s 10 -w 5`

3. Run MPI test cases (multi-rank):
   - `mpirun -np 2 ./build-release/main/src/sphexa/sphexa --init sedov -n 50 -s 10 -w 5`
   - `mpirun -np 4 ./build-release/main/src/sphexa/sphexa --init sedov -n 50 -s 10 -w 5`

4. Run analytical solution comparisons (if available):
   - Check `main/src/analytical_solutions/` for comparison scripts

5. Report:
   - Pass/fail per test case
   - Any crashes or numerical issues
   - Performance characteristics (particles/sec)

If ANY test case crashes or produces NaN values, investigate before declaring complete.

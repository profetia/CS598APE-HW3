# CS598APE HW3

This repository contains our optimized implementation of the Abelian sandpile
identity computation for CS598APE Homework 3.

![Identity sandpile on a 1001x1001 pacman grid](identity-ref.png)

## 1. Build

The artifact requires Git, Python 3, and a GCC installation with OpenMP and
AVX-512F support. Build the program from the repository root with:

```bash
make clean
make -j4
```

Run the target workload on four threads with:

```bash
OMP_NUM_THREADS=4 ./sandpile.exe 1001
```

The generated `identity.png` should match the reference output:

```bash
sha256sum identity.png
# 4883dba5b09236b78477ca25c9395411fc1f82f4233aaa378193d7925269dbbe
```

## 2. Benchmark

`benchmark.py` builds each requested commit in an isolated Git worktree, runs
it five times by default, verifies the output hash, and reports the minimum and
mean runtime. Omitting `--commits` benchmarks the current working tree. For
example:

```bash
./benchmark.py --commits 392d6a5 0f4466a --runs 3 --csv
```

Results and optional CSV files are written below `runs/`.

## 3. Evaluation

The following commands reproduce the measurements presented in the report.
All commands use a grid size of 1001 and four OpenMP threads unless overridden.

### Overall Performance

The main figure compares the baseline, O1 precomputed row bounds, O2 OpenMP
stabilization, O3 AVX-512 vectorization, and O4 reflection symmetry:

```bash
./benchmark.py \
  --commits 15fe837 392d6a5 0f4466a 08e7e49 8195a9b \
  --csv
```

### OpenMP Strip Partitioning

The breakdown figure uses two instrumented commits. The first selects
equal-row or active-cell-balanced strips through `PARTITION`; the second
instruments dynamic 32-row chunks:

```bash
PARTITION=equal ./benchmark.py --commits df3b52e --runs 1
./benchmark.py --commits df3b52e --runs 1
./benchmark.py --commits 6ddc59b --runs 1
```

The profiler prints aggregate time for computation and barrier wait for the four threads.

### SIMD Width

The commits below evaluate scalar, SSE2, AVX2, and AVX-512 implementations, in
that order:

```bash
./benchmark.py \
  --commits df9e84a a44de33 b7d2a05 08e7e49 \
  --runs 3 --csv
```

### Selective Row-Block Scanning

For the rejected selective-scanning experiment, O4 provides the disabled
configuration, followed by block sizes of 1, 8, 16, 32, and 64 rows:

```bash
python3 selective_profile.py --runs 3
```


# Block-Sparse JVP Benchmark Results

## Environment

Measurements were taken on 2026-07-29 using physical GPU 6:

- NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0
- Python 3.11.14
- PyTorch 2.11.0+cu130
- Triton 3.6.0
- bfloat16, 12 heads, head dimension 64
- parent git revision `f7100a84b348b90ecee45707c3205108ef68fecd`

The benchmark metadata records `git_dirty=true` because these measurements
were made against the implementation in this working tree before it was
committed. Compilation, allocator initialization, mask construction, and input
creation were excluded from CUDA-event timing. Primary rows use 25 warmups and
50 measured repetitions; matrix rows use 25 warmups and 30 repetitions.
Dedicated eager and compiled JVP stability checks use 100 warmups and 200
measured repetitions.

The advertised minimum versions were checked in a separate environment without
changing the locked development environment. Python 3.10.19, PyTorch
2.8.0+cu128, Torchvision 0.23.0+cu128, and Triton 3.4.0 passed all 65 non-slow
tests on GPU 6, including eager and compiled pointer/TMA sparse paths. The same
65 tests also passed under Python 3.10.19 with the locked PyTorch 2.11 stack.

`expanded dense` reproduces the current `jit_sandbox` JVP wrapper, including
its per-call expansion of a shared mask to contiguous `[B,H,N,N]` storage.
The harness also records a `broadcast_dense` control that passes the shared 2D
mask directly, isolating dense-kernel time from that materialization.

## Dense/no-mask regression

The contribution was compared directly with parent revision `f7100a8` using
`benchmarks/benchmark_dense_regression.py`. Both runs used bfloat16
`[1,12,3904,64]` Q/K/V tensors, TMA, 100 warmups, and 200 measured fused
primal/JVP calls on GPU 6.

| revision         | execution |   median |        p20-p80 |
| ---------------- | --------- | -------: | -------------: |
| parent `f7100a8` | eager     | 1.589 ms | 1.562-1.625 ms |
| contribution     | eager     | 1.582 ms | 1.546-1.606 ms |
| contribution     | compiled  | 0.796 ms | 0.794-0.798 ms |

The comparable eager path is 0.45% faster than the parent, passing the
no-more-than-5% regression criterion. The parent compiled path cannot serve as
a timing baseline in this environment because Dynamo rejects its in-graph
`triton.set_allocator` call; the contribution's compiled dense path runs
successfully.

## Primary pMF-B result

The primary `hilbert_random`, registers-0 fixture has sequence length 3896
(3904 padded), 10.13% allowed elements, and 11.19% occupied 32x32 tiles.
The sparse metadata is 0.347 MiB, versus 174.4 MiB for the expanded
`[1, 12, 3904, 3904]` Boolean mask: a 503x storage reduction.

### Eager, batch 1

| workload                    | expanded dense | sparse pointer | speedup | memory reduction |
| --------------------------- | -------------: | -------------: | ------: | ---------------: |
| primal forward              |       0.966 ms |       0.161 ms |   6.00x |            7.82x |
| fused primal + JVP          |       1.396 ms |       0.538 ms |   2.60x |            3.18x |
| primal + backward           |       2.384 ms |       0.459 ms |   5.19x |            4.33x |
| fused JVP + primal backward |       2.866 ms |       0.892 ms |   3.21x |            3.18x |

At batch 2, fused JVP improves from 2.570 ms to 0.657 ms (3.91x);
fused JVP plus ordinary backward improves from 5.213 ms to 1.038 ms
(5.02x).

### Compiled

| batch | workload                    | expanded dense | sparse pointer | speedup |
| ----: | --------------------------- | -------------: | -------------: | ------: |
|     1 | fused primal + JVP          |       1.059 ms |       0.249 ms |   4.25x |
|     1 | fused JVP + primal backward |       2.778 ms |       0.445 ms |   6.24x |
|     2 | fused primal + JVP          |       2.013 ms |       0.368 ms |   5.47x |
|     2 | fused JVP + primal backward |       5.368 ms |       0.649 ms |   8.27x |

On the 200-sample eager stability check, TMA is 2.2% slower than the pointer
kernel for fused JVP (0.551 versus 0.539 ms) and 0.3% slower for the pMF
training workload (0.923 versus 0.920 ms). This is within the 10% acceptance
bound. Both routes remain explicit through `USE_TMA`.

## All realistic masks

These geometric means cover `chunk`, `row`, `ring`, `hilbert_4`,
`hilbert_16`, `hilbert_64`, and `hilbert_random` with 0, 1, and 4 register
tokens:

| execution | batch | fused-JVP geomean | worst mask | JVP + backward geomean | worst mask |
| --------- | ----: | ----------------: | ---------: | ---------------------: | ---------: |
| eager     |     1 |             2.76x |      2.48x |                  3.40x |      2.95x |
| eager     |     2 |             4.19x |      3.82x |                  5.33x |      4.78x |
| compiled  |     1 |             4.68x |      4.11x |                  5.55x |      4.45x |
| compiled  |     2 |             5.73x |      5.03x |                  7.84x |      7.39x |

No realistic mask regressed. Every mask is comfortably above the 1.5x
geometric-mean acceptance threshold.

## Density and extended cases

For full 32x32 synthetic tiles, eager fused-JVP speedup versus the expanded
dense path is 3.09x at 5% occupancy, 2.81x at 15%, 1.70x at 50%, 1.35x at
75%, and 1.14x at 100%. Against the broadcast-dense path, sparse remains
faster through 75% occupancy and is about 2% slower at 100%, locating the
crossover between those two points.

The non-gating 256x512 pMF case has sequence length 7800 and 10.44% tile
occupancy. Fused JVP improves from 4.972 ms to 0.826 ms (6.02x); fused JVP
plus backward improves from 10.313 ms to 1.392 ms (7.41x), with a 5.75x
incremental-memory reduction.

Compiled FlexAttention is included only as primal context: 0.084 ms forward
and 0.332 ms forward plus backward on the primary mask. It is not a direct
baseline because it cannot calculate the required JVP.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=6 uv run python benchmarks/benchmark_block_sparse.py \
  --batch 1 2 --warmup 25 --repeats 50

CUDA_VISIBLE_DEVICES=6 uv run python benchmarks/benchmark_block_sparse.py \
  --matrix --batch 1 2 --workloads dual dual_backward \
  --warmup 25 --repeats 30

CUDA_VISIBLE_DEVICES=6 uv run python benchmarks/benchmark_block_sparse.py \
  --compiled --batch 1 2 --warmup 25 --repeats 50

CUDA_VISIBLE_DEVICES=6 uv run python benchmarks/benchmark_block_sparse.py \
  --workloads dual dual_backward --warmup 100 --repeats 200

CUDA_VISIBLE_DEVICES=6 uv run python benchmarks/benchmark_block_sparse.py \
  --compiled --workloads dual dual_backward --warmup 100 --repeats 200

CUDA_VISIBLE_DEVICES=6 uv run python benchmarks/benchmark_block_sparse.py \
  --density-sweep --workloads dual --warmup 25 --repeats 30

CUDA_VISIBLE_DEVICES=6 uv run python benchmarks/benchmark_block_sparse.py \
  --extended --workloads dual dual_backward --warmup 25 --repeats 30

CUDA_VISIBLE_DEVICES=6 uv run python benchmarks/benchmark_block_sparse.py \
  --include-flex --workloads primal primal_backward \
  --warmup 25 --repeats 30

CUDA_VISIBLE_DEVICES=6 uv run python benchmarks/benchmark_dense_regression.py \
  --warmup 100 --repeats 200

CUDA_VISIBLE_DEVICES=6 uv run python benchmarks/benchmark_dense_regression.py \
  --compiled --warmup 100 --repeats 200
```

The harness writes Markdown and JSON with raw samples, medians, p20/p80,
minimums, peak allocated/reserved memory, mask statistics, and environment
metadata under the ignored `benchmarks/results/` directory.

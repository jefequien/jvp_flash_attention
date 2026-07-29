# Block-Sparse JVP Flash Attention

## Status

Implemented and validated on 2026-07-29. All implementation, tests, fixtures,
and benchmarks live in this repository and do not depend on a sibling
`jit_sandbox` checkout. See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for
the measured acceptance results.

The development environment is pinned with uv to match the current
`jit_sandbox` kernel environment:

- Python 3.11
- PyTorch 2.11.0+cu130
- Torchvision 0.26.0+cu130
- Triton 3.6.0
- pytest and Ruff as development dependencies

Set up or refresh the environment with:

```bash
uv sync --python 3.11 --group dev --extra lint
```

## Objective

Add an opt-in block-sparse Boolean mask path to JVP Flash Attention which:

1. skips Q/KV tiles that are completely masked;
2. preserves exact element-level masking inside partially masked tiles;
3. supports the fused primal and JVP forward calculation;
4. supports the ordinary reverse-mode backward calculation needed for model
   training;
5. works through both direct forward-AD dual tensors and `torch.func.jvp`;
6. works under `torch.compile`;
7. removes the need to materialize a dense `[batch, heads, sequence, sequence]`
   Boolean mask for masks shared by batches and heads; and
8. produces a meaningful speed and memory improvement for the block-causal pMF
   masks used by `jit_sandbox`.

The primary target is square self-attention with a head dimension of 64,
bfloat16 inputs, and sequence lengths around 3,800-5,000 tokens. This is the
hot path for the current pixel pMF-B configuration.

## Why this belongs in this repository

The current forward and JVP kernel already performs an online-softmax reduction
over 32x32 Q/KV tiles. Block sparsity changes the tile schedule, not the JVP
formula. The backward implementation already has the two traversal directions
needed by a sparse implementation:

- dQ reduces over KV tiles for a fixed Q tile;
- dK/dV reduce over Q tiles for a fixed KV tile.

A sparse mask therefore needs both a row-oriented schedule and its transpose.
This is the same information carried by PyTorch FlexAttention's `BlockMask`,
but the JVP kernels and all supporting code will remain implemented here.

## Scope

### Included

- Boolean block-sparse masks.
- Square Q/K/V self-attention, matching the existing JVP kernel contract.
- Mask batch/head dimensions of either `1` or the corresponding Q dimension,
  including a single mask shared across all batches and heads.
- Internal padding to the kernel tile size, with outputs sliced back to the
  original sequence length.
- float16, bfloat16, and float32.
- Existing supported head dimensions: 16, 32, 64, 128, and 256.
- Non-TMA and TMA forward/JVP paths.
- Reverse-mode dQ, dK, and dV.
- Eager execution and `torch.compile`.
- CUDA correctness and performance on the available Blackwell system.
- A representation that can be constructed from a dense Boolean mask and,
  optionally, from the public tensor payload of a PyTorch `BlockMask`.

### Not included in the first implementation

- Sparse additive attention biases.
- Reverse-mode differentiation through the JVP tangent; the pMF path detaches
  the tangent and requires ordinary reverse mode through the primal only.
- Attention dropout, which is already unsupported by the public JVP path.
- Rectangular cross-attention.
- Sparse FP8 validation or tuning.
- A callable `score_mod` or arbitrary Python `mask_mod` inside the Triton
  kernel.
- Changes to `jit_sandbox`. Integration there happens only after this
  repository has a validated release/API.
- A ROCm performance gate. The implementation should avoid gratuitous
  CUDA-only assumptions, but ROCm remains unverified without suitable hardware.

Dense Boolean masks, additive masks, no-mask attention, and the existing causal
path must remain available and backward compatible.

## Proposed public API and mask contract

Add a tensor-only, pytree-compatible mask type in
`jvp_flash_attention/block_sparse.py`, provisionally named
`BlockSparseMask`.

The intended user flow is:

```python
block_mask = BlockSparseMask.from_bool(
    boolean_mask,
    block_size=32,
)

output = JVPAttn.fwd_dual(
    q,
    k,
    v,
    block_mask=block_mask,
)
```

`attn_mask` and `block_mask` are mutually exclusive. Existing calls using
`attn_mask` must behave as before.

`BlockSparseMask` should contain only tensors plus small immutable scalar
metadata so that it can be cached, moved between devices, flattened through
PyTorch transforms, and passed through the custom compiled operator.

The representation should contain:

- original and padded sequence lengths;
- block size, initially required to be 32x32;
- per-row counts and padded column-index arrays for partial KV tiles;
- per-row counts and padded column-index arrays for full KV tiles;
- the transposed partial and full schedules used by dK/dV;
- compact element-level Boolean masks for partial tiles;
- edge/mask IDs connecting both traversal directions to the same partial tile;
  and
- batch/head broadcast metadata.

Construction must classify every padded tile into exactly one of:

- empty: omit it from all schedules;
- full: schedule it without loading an element mask; or
- partial: schedule it and load its compact 32x32 Boolean mask.

For internal padding, real query rows must not attend to padded keys. Each
padded query row should attend only to itself so that no softmax row is empty.
The returned output is sliced to the original query length.

The constructor should reject a real query row with no visible key. This makes
the behavior explicit rather than relying on an undefined all-masked softmax.

An optional `BlockSparseMask.from_flex(...)` adapter may consume the public
tensor metadata and evaluate `mask_mod` once during preprocessing. Attention
calls must never invoke `mask_mod` or recompress the mask per layer.

## Implementation plan

### Phase 0: Make the test and benchmark harness dependable

- Split correctness tests from the current monolithic benchmark script.
- Keep the existing script working for compatibility.
- Fix the `--no-benchmark-performance` path, which currently references `np`
  when matplotlib/numpy plotting support is unavailable.
- Add a `cuda` pytest marker and make CUDA tests skip cleanly when no GPU is
  visible.
- Add shared helpers for deterministic Q/K/V primals, tangents, output
  cotangents, timing, and memory measurement.
- Record software versions, device name, compute capability, dtype, shapes,
  seed, and git revision in benchmark output.

Exit criteria:

- `uv run pytest -m "not slow"` collects real tests and exits successfully.
- The existing dense bfloat16 length-32 correctness matrix passes.
- Running correctness without performance or plotting dependencies succeeds.

### Phase 1: Sparse mask construction and validation

- Implement `BlockSparseMask.from_bool`.
- Implement padding, full/partial/empty classification, compact partial masks,
  and transposed schedules.
- Register the mask as a pytree if a dataclass alone is insufficient for
  `torch.compile`.
- Add `.to(device)`, validation, useful shape/sparsity properties, and a
  debugging reconstruction method.
- Support masks shaped `[Q,K]`, `[1,1,Q,K]`, `[B,1,Q,K]`, `[1,H,Q,K]`, and
  `[B,H,Q,K]`.
- Cache no derived tensors inside an attention call. Construction is explicitly
  outside the timed/model hot path.

Exit criteria:

- Reconstructing the dense padded mask from the sparse representation is
  exactly equal to the expected Boolean mask.
- Forward and transposed schedules contain the same edges.
- Every scheduled tile is in bounds, unique within its row, and correctly
  classified.
- Serialized tensor storage for the default pMF fixture is below 2 MiB and is
  at least 50x smaller than its expanded `[1,12,N,N]` dense mask.

### Phase 2: Non-TMA sparse primal and JVP forward

- Add a sparse forward inner loop which iterates only scheduled full and partial
  KV tiles.
- Reuse the existing online max, normalization, primal accumulation, and JVP
  accumulation.
- Load and apply element masks only for partial tiles.
- Preserve the dense forward path rather than adding sparse conditionals to
  every dense tile.
- Thread sparse tensors and scalar metadata through `JVPAttn.forward`,
  `JVPAttn.fwd`, and `JVPAttn.fwd_dual`.
- Extend the opaque `torch.library.custom_op` and fake implementation used by
  the compiled dual forward.
- Initially run with `USE_TMA=False` to isolate scheduling correctness.

Exit criteria:

- Primal and tangent results pass the numerical matrix below.
- Direct forward-AD and `torch.func.jvp` agree.
- The compiled and eager sparse outputs agree.
- A mask containing both full and partial tiles is covered by every transform
  test.
- Dense/no-mask behavior is unchanged.

### Phase 3: Sparse reverse-mode backward

- Modify dQ to consume the sparse KV-row schedule.
- Modify dK/dV to consume the transposed Q-row schedule.
- Reconstruct probabilities from the saved log-sum-exp exactly as the dense
  backward does.
- Apply compact element masks only for partial tiles.
- Keep one program responsible for each output gradient tile so no atomics are
  introduced.
- Ensure mask tensors are treated as nondifferentiable constants.

Exit criteria:

- dQ, dK, and dV pass the numerical matrix below.
- No gradient contains NaN or infinity.
- Backward works after both an ordinary sparse forward and a
  `torch.func.jvp` forward whose tangent is detached in the pMF style.
- Backward works under `torch.compile`.

### Phase 4: TMA support and kernel tuning

- Add sparse schedules to `_attn_fwd_tma`.
- Verify tensor descriptor loads at non-contiguous scheduled KV block offsets.
- Compare separate full/partial loops against an ordered combined loop if
  numerical order or branch overhead is material.
- Tune block traversal and launch configuration for head dimensions 64 and 128
  on the available Blackwell GPU.
- Keep block size 32 as the correctness baseline. Only add another sparse block
  size if a benchmark demonstrates a repeatable benefit without losing partial
  mask fidelity.

Exit criteria:

- `USE_TMA=True` and `USE_TMA=False` pass the same correctness matrix on
  supported hardware.
- TMA is not slower than non-TMA by more than 10% on the default pMF benchmark,
  or the faster path is selected explicitly and the reason is documented.

### Phase 5: Documentation and release readiness

- Document construction, caching, padding, broadcast semantics, limitations,
  and examples in the README.
- Document that mask preprocessing is excluded from per-layer timing and should
  be cached by callers.
- Add a changelog/release note if the repository adopts one.
- Retain the dense API and all existing examples.

Exit criteria:

- A user can reproduce correctness and benchmark results from a fresh checkout
  with only `uv sync` and the documented commands.
- No test or benchmark imports `jit_sandbox` or reads outside this repository.

## Correctness test plan

### Reference implementation

For small and medium cases, use a pure PyTorch reference:

1. compute scaled QK scores in float32;
2. apply the Boolean mask with negative infinity;
3. apply softmax;
4. multiply by V; and
5. use `torch.func.jvp` and `torch.autograd.grad` on this explicit function.

Also compare the sparse kernel directly with the existing dense JVP kernel.
This separates sparse scheduling errors from the numerical error already
present in the Triton formulation.

### Mask representation unit tests

Create `tests/test_block_sparse.py` covering:

- all-full mask;
- ordinary causal mask;
- block-diagonal mask;
- sliding-window block-causal mask;
- randomly selected full tiles;
- a mask with one partial tile;
- partial first and last tiles;
- non-multiple-of-32 lengths;
- batch-only, head-only, and batch/head-specific masks;
- deterministic construction;
- `.to(device)` and pytree flatten/unflatten;
- dense reconstruction;
- forward/transposed edge equivalence;
- invalid shape, dtype, block size, and empty real query rows; and
- optional equivalence with a PyTorch FlexAttention `BlockMask`.

### Kernel unit tests

Create `tests/test_sparse_attention.py` and
`tests/test_sparse_attention_slow.py` with parameterized cases over:

- dtype: float32, float16, bfloat16;
- head dimension: 16, 32, 64, 128, 256;
- sequence length: 32, 60, 64, 96, 127, 128, 257;
- batch/head shape: `(1,1)`, `(2,1)`, `(1,4)`, `(2,4)`;
- mask pattern: full, causal, block diagonal, sliding window, random full
  blocks, mixed full/partial, and pMF miniature;
- tangent source: Q only, K only, V only, and Q/K/V together;
- execution: eager and compiled; and
- `USE_TMA`: false and true where supported.

The full Cartesian product is unnecessary. Define a covering matrix so every
value and interaction is exercised without making the default test suite
prohibitively slow. Mark the full dtype/head/shape sweep as `slow`.

For every selected case, check:

- primal output;
- JVP tangent;
- dQ, dK, and dV for a deterministic random output cotangent;
- direct `fwAD.make_dual` versus `torch.func.jvp`;
- sparse versus dense JVP kernel;
- sparse versus explicit PyTorch reference;
- finite outputs and gradients; and
- masked logits never affect the online-softmax maximum, including partial
  tiles with no visible key for an individual row; and
- internal padding does not affect real-token outputs.

Add a small float32 central finite-difference test for the JVP tangent. This is
an independent check of the reference-transform path.

### pMF-style training test

Add a test which mirrors the relevant pMF differentiation structure:

```python
primal, tangent = torch.func.jvp(attention_fn, primals, tangents)
compound = primal + scale * tangent.detach()
loss = compound.float().square().mean()
loss.backward()
```

Compare the loss and Q/K/V gradients with the dense JVP kernel and explicit
reference. This test is required because pMF needs the fused JVP forward plus
ordinary primal backward, not a backward through the tangent.

### `jit_sandbox` mask fixtures implemented locally

Create `tests/jit_sandbox_masks.py` containing a small, dependency-free
reproduction of the current packed pMF mask semantics. It should accept a
configuration object containing:

- image height and width;
- target frames;
- temporal and spatial patch sizes;
- block mode;
- register tokens per block;
- causal history in temporal chunks; and
- deterministic random seed.

The default realistic fixture is:

- image: 256x256;
- target frames: 120;
- one observed condition chunk;
- patch size: `(4, 32, 32)`;
- token grid: 31x8x8 including the condition chunk;
- history: 6 chunks;
- block mode: `hilbert_random`;
- block-size choices: 4, 8, 16, 32, or 64 independently per temporal chunk;
- seed: 0;
- registers: 0 by default, with 1 and 4 as ablations; and
- one shared mask across batch and attention heads.

The fixture must reproduce clean/noisy stream packing, fixed observed tokens,
the omission of the final clean block, block-causal visibility, same-token
exclusions, and register placement. Add semantic spot tests for each rule.

Expected fixture statistics provide regression checks:

| Mode/configuration            | Sequence | Padded | Element density | Nonempty 32x32 tiles | Full among nonempty |
| ----------------------------- | -------: | -----: | --------------: | -------------------: | ------------------: |
| `chunk`, registers 0          |     3840 |   3840 |          10.67% |               10.67% |             100.00% |
| `row`, registers 0            |     3896 |   3904 |           9.81% |               10.51% |              83.51% |
| `ring`, registers 0           |     3876 |   3904 |          10.02% |               11.93% |              73.59% |
| `hilbert_4`, registers 0      |     3900 |   3904 |           9.75% |               10.51% |              83.51% |
| `hilbert_16`, registers 0     |     3888 |   3904 |           9.93% |               10.51% |              83.51% |
| `hilbert_64`, registers 0     |     3840 |   3840 |          10.67% |               10.67% |             100.00% |
| `hilbert_random`, registers 0 |     3896 |   3904 |          10.13% |               11.19% |              81.27% |
| `hilbert_random`, registers 1 |     4168 |   4192 |          10.06% |               12.10% |              68.83% |
| `hilbert_random`, registers 4 |     4984 |   4992 |           9.91% |               11.54% |              74.36% |

Statistics should be checked with exact counts internally; rounded percentages
are shown here for readability.

### Numerical acceptance

Use `torch.testing.assert_close` with dtype-specific tolerances, defined once in
the tests. Initial upper bounds are:

- float32 primal: `atol=3e-4`, `rtol=3e-4`;
- float32 tangent/gradients: `atol=1e-3`, `rtol=1e-3`;
- float16 primal: `atol=1e-2`, `rtol=1e-2`;
- float16 tangent/gradients: `atol=2e-2`, `rtol=2e-2`;
- bfloat16 primal: `atol=2e-2`, `rtol=2e-2`; and
- bfloat16 tangent/gradients: `atol=4e-2`, `rtol=4e-2`.

For the sparse-versus-explicit comparison, also record the existing dense
kernel's error against the same reference. Sparse error must not exceed the
dense error by more than 25% plus the dtype's absolute tolerance. Tolerances may
only be relaxed after recording a failing case and explaining why the reference
or accumulation order requires it.

The implemented float32 covering matrix uses `atol=rtol=2e-2` because Triton
uses TF32 tensor-core dots on the tested NVIDIA build. A separate float32
central-difference test guards the JVP formula independently of that
accumulation difference.

## Regression acceptance criteria

The implementation is correct only if all of the following hold:

- All newly added unit tests pass.
- All existing dense correctness configurations continue to pass at their
  current tolerances.
- Existing public calls need no source changes.
- Dense Boolean masks, additive masks, no-mask attention, and ordinary causal
  attention retain their behavior.
- Dropout continues to fail explicitly rather than silently changing behavior.
- `uv run pytest -m "not slow"` passes from a fresh environment.
- `uv run pytest -m slow` passes on a CUDA host with sufficient memory.
- Ruff reports no new violations in touched or added files.
- No sparse call constructs an `O(sequence^2 * batch * heads)` tensor.
- Sparse-mask preprocessing occurs once in the benchmark fixture, outside the
  timed attention calls.

## Benchmark plan

### Benchmark entry point

Add `benchmarks/benchmark_block_sparse.py`. It must run independently from
`jit_sandbox` and emit:

- a human-readable Markdown table;
- machine-readable JSON;
- raw median and percentile timings;
- peak allocated and reserved memory;
- mask element density and tile occupancy;
- software/device/git metadata; and
- speedup and memory ratios against the dense kernel.

Generated results should be ignored by git unless a specific reviewed baseline
is intentionally checked in.

### Compared implementations

Measure:

1. existing dense-mask JVP Flash Attention;
2. broadcast-dense JVP Flash Attention, after 2D mask broadcasting is added;
3. new block-sparse JVP Flash Attention;
4. explicit PyTorch math attention for small correctness cases only; and
5. FlexAttention primal forward/backward as contextual sparse lower-bound
   numbers, clearly labeled as not supporting JVP and therefore not a direct
   speedup baseline.

The primary reported speedup is dense JVP Flash Attention divided by sparse JVP
Flash Attention for identical primal/tangent work.

### Timed workloads

Time these workloads separately:

- primal forward;
- fused primal plus JVP forward;
- ordinary primal forward plus reverse backward;
- fused primal/JVP forward plus ordinary reverse backward, matching pMF;
- eager steady state; and
- compiled steady state.

Compilation, mask construction, random tensor creation, and allocator warm-up
must be excluded from kernel timing. Report mask construction time separately
because callers need to understand its amortization.

The `expanded_dense` end-to-end baseline intentionally retains
`jit_sandbox`'s current per-call `[B,H,N,N]` expansion. The
`broadcast_dense` row excludes that materialization and serves as the
kernel-only dense control.

Use CUDA events or `triton.testing.do_bench`, explicit synchronization,
deterministic inputs, at least 25 warm-up iterations, and enough repetitions for
a stable median. Report median, 20th percentile, and 80th percentile rather
than only a mean.

Peak memory must be measured after warm-up by recording the allocation before
the operation and subtracting it from `torch.cuda.max_memory_allocated()`.

### Realistic benchmark matrix

Primary shape:

- dtype: bfloat16;
- batch: 1 and 2;
- heads: 12;
- head dimension: 64;
- pMF-B mask geometry from the local fixture;
- history: 6 chunks; and
- eager and compiled execution.

Required masks:

- `chunk`;
- `row`;
- `ring`;
- `hilbert_4`;
- `hilbert_16`;
- `hilbert_64`;
- `hilbert_random` with seed 0;
- `hilbert_random` with 1 register per block; and
- `hilbert_random` with 4 registers per block.

Add an extended, non-gating 256x512 case to represent the wider supported
video datasets. Add a small density sweep to identify the occupancy at which
the sparse path stops outperforming dense attention.

### Recorded pre-implementation baseline

On an NVIDIA RTX PRO 6000 Blackwell Server Edition with the uv environment
listed above, the default `hilbert_random`, registers-0 fixture produced:

- sequence length 3896, padded to 3904;
- 10.13% allowed elements;
- 11.19% nonempty 32x32 tiles;
- 81.27% full tiles among the nonempty tiles;
- approximately 174.4 MiB for a padded Boolean mask expanded to
  `[1,12,3904,3904]`;
- approximately 382.7 MiB incremental peak allocation for a current dual call;
- 2.91 ms for the current dense fused primal/JVP forward; and
- 4.46 ms for the current dense fused forward plus ordinary reverse backward.

These are diagnostic baselines, not permanent golden timings. The benchmark
script must regenerate them using controlled methodology.

### Performance acceptance

Performance is evaluated manually on the named Blackwell system, not asserted
inside ordinary unit tests.

For bfloat16, batch 1, 12 heads, head dimension 64, and the default realistic
`hilbert_random` mask:

- sparse fused primal/JVP forward must be at least 1.5x faster than the current
  expanded-dense-mask path;
- sparse fused forward plus ordinary backward must be at least 1.5x faster;
- incremental peak allocated memory must be at least 2x lower; and
- sparse mask storage must be at least 50x smaller than the expanded dense
  mask.

Across all required realistic masks:

- the geometric-mean speedup for fused forward/JVP must be at least 1.5x;
- no mask with at most 15% nonempty 32x32 tiles may regress by more than 10%;
  and
- any regression must include raw measurements and a documented explanation
  before acceptance.

The existing dense/no-mask path must not regress by more than 5% in its own
steady-state benchmark.

## Completion checklist

- [x] Test harness is deterministic and works without plotting dependencies.
- [x] `BlockSparseMask` and validation are implemented.
- [x] Sparse non-TMA primal/JVP forward is implemented.
- [x] Sparse dQ/dK/dV are implemented.
- [x] Compiled custom-op and fake paths carry sparse metadata.
- [x] Sparse TMA forward is implemented and selected appropriately.
- [x] Dense reconstruction and schedule invariant tests pass.
- [x] Primal, tangent, and gradient correctness matrices pass.
- [x] pMF-style detached-tangent training test passes.
- [x] Local realistic `jit_sandbox` mask fixtures pass semantic regression tests.
- [x] Realistic eager and compiled benchmarks are reproducible.
- [x] Performance and memory acceptance thresholds are met.
- [x] Existing dense APIs and tests remain green.
- [x] README documents the new API, caching, limitations, and commands.

"""Benchmark block-sparse JVP attention with realistic pMF masks.

Example:

    CUDA_VISIBLE_DEVICES=6 uv run python benchmarks/benchmark_block_sparse.py \
        --warmup 25 --repeats 100
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess  # nosec B404 - used only for fixed, read-only git metadata commands
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask as FlexBlockMask
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)

from benchmarks.pmf_masks import PMFMaskConfig, make_pmf_mask_from_config
from jvp_flash_attention import BlockSparseMask, JVPAttn

MATRIX_CONFIGS = (
    ("chunk", 0),
    ("row", 0),
    ("ring", 0),
    ("hilbert_4", 0),
    ("hilbert_16", 0),
    ("hilbert_64", 0),
    ("hilbert_random", 0),
    ("hilbert_random", 1),
    ("hilbert_random", 4),
)
DENSITY_VALUES = (0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.00)


@dataclass(frozen=True)
class Timing:
    median_ms: float
    p20_ms: float
    p80_ms: float
    minimum_ms: float
    samples_ms: list[float]


@dataclass(frozen=True)
class Memory:
    peak_allocated_mib: float
    peak_reserved_mib: float


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _time_cuda(
    function: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
) -> Timing:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del result
    ordered = sorted(samples)
    return Timing(
        median_ms=_percentile(ordered, 0.5),
        p20_ms=_percentile(ordered, 0.2),
        p80_ms=_percentile(ordered, 0.8),
        minimum_ms=ordered[0],
        samples_ms=samples,
    )


def _memory_cuda(function: Callable[[], object]) -> Memory:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base_allocated = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    result = function()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del result
    return Memory(
        peak_allocated_mib=max(0, peak_allocated - base_allocated) / 2**20,
        peak_reserved_mib=max(0, peak_reserved - base_reserved) / 2**20,
    )


def _git_revision() -> str:
    try:
        return subprocess.run(  # nosec B603, B607
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool | None:
    try:
        status = subprocess.run(  # nosec B603, B607
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(status.strip())


def _metadata(args: argparse.Namespace) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "total_device_memory_mib": properties.total_memory / 2**20,
        "dtype": str(args.dtype).removeprefix("torch."),
        "heads": args.heads,
        "head_dim": args.head_dim,
        "batch": args.batch,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "compiled": args.compiled,
    }


def _make_workload(
    implementation: str,
    workload: str,
    *,
    primals: tuple[Tensor, Tensor, Tensor],
    tangents: tuple[Tensor, Tensor, Tensor],
    output_cotangent: Tensor,
    dense_mask: Tensor,
    block_mask: BlockSparseMask,
    flex_block_mask: FlexBlockMask | None,
    use_compile: bool,
) -> Callable[[], object]:
    is_sparse = implementation.startswith("sparse")
    use_tma = implementation == "sparse_tma"
    if use_compile or implementation == "flex":
        # Each row owns its compiled callable and is warmed before timing.
        # Resetting here prevents unrelated shapes/closures in a benchmark
        # matrix from consuming Dynamo's per-code-object recompile budget.
        torch._dynamo.reset()
    compiled_flex = (
        torch.compile(flex_attention, fullgraph=True) if implementation == "flex" else None
    )

    def primal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if implementation == "flex":
            if flex_block_mask is None:
                raise RuntimeError("FlexAttention requires a precomputed BlockMask.")
            assert compiled_flex is not None
            return compiled_flex(q, k, v, block_mask=flex_block_mask)
        if is_sparse:
            return JVPAttn.fwd(
                q,
                k,
                v,
                block_mask=block_mask,
                USE_TMA=use_tma,
            )
        runtime_mask = dense_mask
        if implementation == "expanded_dense":
            # Match jit_sandbox's current JVP wrapper, which expands a shared
            # mask to [B,H,N,N] inside every attention call.
            runtime_mask = dense_mask.expand(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                q.shape[2],
            ).contiguous()
        return JVPAttn.fwd(
            q,
            k,
            v,
            attn_mask=runtime_mask,
            USE_TMA=False,
        )

    def dual_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if implementation == "flex":
            raise RuntimeError("FlexAttention does not support the JVP workload.")
        if is_sparse:
            return JVPAttn.fwd_dual(
                q,
                k,
                v,
                block_mask=block_mask,
                USE_TMA=use_tma,
            )
        runtime_mask = dense_mask
        if implementation == "expanded_dense":
            runtime_mask = dense_mask.expand(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                q.shape[2],
            ).contiguous()
        return JVPAttn.fwd_dual(
            q,
            k,
            v,
            attn_mask=runtime_mask,
            USE_TMA=False,
        )

    primal_callable = (
        torch.compile(primal_attention, fullgraph=True)
        if use_compile and implementation != "flex"
        else primal_attention
    )

    def fused_dual(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        tangent_q: Tensor,
        tangent_k: Tensor,
        tangent_v: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return torch.func.jvp(
            dual_attention,
            (q, k, v),
            (tangent_q, tangent_k, tangent_v),
        )

    dual_callable = torch.compile(fused_dual, fullgraph=True) if use_compile else fused_dual

    if workload == "primal":
        return lambda: primal_callable(*primals)
    if workload == "dual":
        return lambda: dual_callable(*primals, *tangents)

    grad_primals = tuple(tensor.detach().requires_grad_() for tensor in primals)
    if workload == "primal_backward":

        def primal_backward() -> tuple[Tensor, ...]:
            output = primal_callable(*grad_primals)
            return torch.autograd.grad(output, grad_primals, output_cotangent)

        return primal_backward
    if workload == "dual_backward":

        def dual_backward() -> tuple[Tensor, ...]:
            primal, tangent = dual_callable(*grad_primals, *tangents)
            # This mirrors pMF: reverse mode traverses the primal only.
            del tangent
            return torch.autograd.grad(primal, grad_primals, output_cotangent)

        return dual_backward
    raise ValueError(f"Unknown workload {workload!r}.")


def _parse_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as error:
        raise argparse.ArgumentTypeError(f"Unknown torch dtype {name!r}.") from error
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise argparse.ArgumentTypeError("dtype must be float16, bfloat16, or float32.")
    return dtype


def _configurations(
    args: argparse.Namespace,
) -> list[tuple[str, int, float | None]]:
    if args.density_sweep:
        return [(f"density_{density:.2f}", 0, density) for density in DENSITY_VALUES]
    if args.matrix:
        return [(mode, registers, None) for mode, registers in MATRIX_CONFIGS]
    return [(args.mode, args.registers, None)]


def _implementations(args: argparse.Namespace) -> tuple[str, ...]:
    implementations = ["expanded_dense", "broadcast_dense", "sparse", "sparse_tma"]
    if args.skip_expanded_dense:
        implementations.remove("expanded_dense")
    if args.include_flex:
        implementations.append("flex")
    return tuple(implementations)


def _density_mask(
    sequence: int,
    density: float,
    *,
    seed: int,
) -> Tensor:
    """Create a deterministic full-tile mask for crossover measurements."""
    if sequence <= 0 or sequence % 32:
        raise ValueError("--density-sequence must be positive and divisible by 32.")
    blocks = sequence // 32
    generator = torch.Generator(device="cuda").manual_seed(seed)
    occupied = (
        torch.rand(
            blocks,
            blocks,
            generator=generator,
            device="cuda",
        )
        < density
    )
    occupied.diagonal().fill_(True)
    return occupied.repeat_interleave(32, 0).repeat_interleave(32, 1)


def _flex_block_mask(mask: Tensor) -> FlexBlockMask:
    """Precompute FlexAttention metadata for contextual primal measurements."""

    def mask_mod(
        _batch: Tensor,
        _head: Tensor,
        query: Tensor,
        key: Tensor,
    ) -> Tensor:
        return mask[query, key]

    compiled_create = torch.compile(create_block_mask, fullgraph=True)
    return compiled_create(
        mask_mod,
        None,
        None,
        mask.shape[-2],
        mask.shape[-1],
        device=mask.device,
        # FlexAttention's current Blackwell kernel requires its public
        # BlockMask tile to be at least 128. mask_mod still preserves exact
        # element-level pMF semantics inside partial Flex tiles.
        BLOCK_SIZE=128,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    torch.manual_seed(args.seed)
    results: list[dict[str, object]] = []
    mask_records: list[dict[str, object]] = []

    for mode, registers, density in _configurations(args):
        fixture_start = time.perf_counter()
        if density is None:
            image_width = 512 if args.extended else args.image_width
            config = PMFMaskConfig(
                image_height=args.image_height,
                image_width=image_width,
                target_frames=args.target_frames,
                temporal_patch=args.temporal_patch,
                spatial_patch_height=args.spatial_patch_height,
                spatial_patch_width=args.spatial_patch_width,
                condition_chunks=args.condition_chunks,
                mode=mode,
                register_tokens_per_block=registers,
                history_chunks=args.history_chunks,
                seed=args.seed,
            )
            fixture = make_pmf_mask_from_config(config, device="cuda")
            dense_boolean_mask = fixture.mask
            sequence = fixture.sequence_length
        else:
            dense_boolean_mask = _density_mask(
                args.density_sequence,
                density,
                seed=args.seed,
            )
            sequence = dense_boolean_mask.shape[-1]
        torch.cuda.synchronize()
        fixture_seconds = time.perf_counter() - fixture_start
        compression_start = time.perf_counter()
        block_mask = BlockSparseMask.from_bool(dense_boolean_mask)
        torch.cuda.synchronize()
        compression_seconds = time.perf_counter() - compression_start
        flex_compression_seconds = None
        if args.include_flex:
            flex_start = time.perf_counter()
            flex_mask = _flex_block_mask(dense_boolean_mask)
            torch.cuda.synchronize()
            flex_compression_seconds = time.perf_counter() - flex_start
        else:
            flex_mask = None

        padded_dense = block_mask.to_dense(padded=True)
        padded_sequence = block_mask.padded_length
        occupied_tiles = block_mask.num_full_tiles + block_mask.num_partial_tiles
        full_fraction = block_mask.num_full_tiles / occupied_tiles if occupied_tiles else 0.0
        mask_record = {
            "mode": mode,
            "registers": registers,
            "sequence": sequence,
            "padded_sequence": padded_sequence,
            "element_density": dense_boolean_mask.float().mean().item(),
            "tile_density": block_mask.tile_density,
            "full_fraction_of_occupied": full_fraction,
            "full_tiles": block_mask.num_full_tiles,
            "partial_tiles": block_mask.num_partial_tiles,
            "sparse_storage_mib": block_mask.storage_bytes / 2**20,
            "expanded_dense_storage_mib_b1": (
                args.heads * padded_sequence * padded_sequence / 2**20
            ),
            "fixture_seconds": fixture_seconds,
            "compression_seconds": compression_seconds,
            "flex_compression_seconds": flex_compression_seconds,
        }
        mask_records.append(mask_record)

        for batch in args.batch:
            originals = tuple(
                torch.randn(
                    batch,
                    args.heads,
                    sequence,
                    args.head_dim,
                    device="cuda",
                    dtype=args.dtype,
                )
                for _ in range(6)
            )
            padded = tuple(
                F.pad(tensor, (0, 0, 0, padded_sequence - sequence)) for tensor in originals
            )
            sparse_cotangent = torch.randn_like(originals[0])
            dense_cotangent = F.pad(sparse_cotangent, (0, 0, 0, padded_sequence - sequence))

            for implementation in _implementations(args):
                is_sparse = implementation.startswith("sparse")
                uses_original_sequence = is_sparse or implementation == "flex"
                attention_mask = (
                    padded_dense if implementation == "expanded_dense" else padded_dense[0, 0]
                )

                selected_inputs = originals if uses_original_sequence else padded
                primals = selected_inputs[:3]
                tangents = selected_inputs[3:]
                cotangent = sparse_cotangent if uses_original_sequence else dense_cotangent
                for workload in args.workloads:
                    if implementation == "flex" and workload in {
                        "dual",
                        "dual_backward",
                    }:
                        continue
                    function = _make_workload(
                        implementation,
                        workload,
                        primals=primals,
                        tangents=tangents,
                        output_cotangent=cotangent,
                        dense_mask=attention_mask,
                        block_mask=block_mask,
                        flex_block_mask=flex_mask,
                        use_compile=args.compiled,
                    )
                    # Compile and initialize allocators before measurement.
                    function()
                    torch.cuda.synchronize()
                    timing = _time_cuda(
                        function,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    memory = _memory_cuda(function)
                    results.append(
                        {
                            "mode": mode,
                            "registers": registers,
                            "batch": batch,
                            "implementation": implementation,
                            "workload": workload,
                            **asdict(timing),
                            **asdict(memory),
                        }
                    )
                del attention_mask
            del originals, padded, sparse_cotangent, dense_cotangent
        del padded_dense

    _annotate_comparisons(results)
    payload = {
        "metadata": _metadata(args),
        "masks": mask_records,
        "results": results,
    }
    return payload


def _annotate_comparisons(results: list[dict[str, object]]) -> None:
    baselines: dict[tuple[str, int, int, str], dict[str, object]] = {}
    for row in results:
        key = (row["mode"], row["registers"], row["batch"], row["workload"])
        if row["implementation"] == "expanded_dense":
            baselines[key] = row
    for row in results:
        key = (row["mode"], row["registers"], row["batch"], row["workload"])
        baseline = baselines.get(key)
        if baseline is None:
            row["speedup_vs_expanded"] = None
            row["memory_reduction_vs_expanded"] = None
            continue
        row["speedup_vs_expanded"] = baseline["median_ms"] / row["median_ms"]
        row["memory_reduction_vs_expanded"] = (
            baseline["peak_allocated_mib"] / row["peak_allocated_mib"]
            if row["peak_allocated_mib"]
            else None
        )


def _markdown(payload: dict[str, object]) -> str:
    results = payload["results"]
    assert isinstance(results, list)
    lines = [
        "| mask | regs | B | implementation | workload | median ms | p20 | p80 | peak MiB | speedup | memory reduction |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        speedup = row["speedup_vs_expanded"]
        memory_reduction = row["memory_reduction_vs_expanded"]
        speedup_text = "n/a" if speedup is None else f"{speedup:.2f}x"
        memory_text = "n/a" if memory_reduction is None else f"{memory_reduction:.2f}x"
        lines.append(
            f"| {row['mode']} | {row['registers']} | {row['batch']} | "
            f"{row['implementation']} | {row['workload']} | "
            f"{row['median_ms']:.3f} | {row['p20_ms']:.3f} | "
            f"{row['p80_ms']:.3f} | {row['peak_allocated_mib']:.1f} | "
            f"{speedup_text} | {memory_text} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="hilbert_random")
    parser.add_argument("--registers", type=int, default=0)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--density-sweep", action="store_true")
    parser.add_argument("--density-sequence", type=int, default=3904)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--target-frames", type=int, default=120)
    parser.add_argument("--temporal-patch", type=int, default=4)
    parser.add_argument("--spatial-patch-height", type=int, default=32)
    parser.add_argument("--spatial-patch-width", type=int, default=32)
    parser.add_argument("--condition-chunks", type=int, default=1)
    parser.add_argument("--history-chunks", type=int, default=6)
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=("primal", "dual", "primal_backward", "dual_backward"),
        default=["primal", "dual", "primal_backward", "dual_backward"],
    )
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument(
        "--include-flex",
        action="store_true",
        help="Include compiled FlexAttention primal-only context numbers.",
    )
    parser.add_argument("--skip-expanded-dense", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/block_sparse.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    markdown = _markdown(payload)
    print(markdown)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    args.output_json.with_suffix(".md").write_text(markdown + "\n")
    print(f"\nWrote {args.output_json} and {args.output_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()

"""Measure the dense/no-mask JVP path without sparse benchmark setup.

Run this file from both the contribution and its base revision with identical arguments.
Compilation and input creation are excluded from timing.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import statistics
from pathlib import Path

import torch
from torch import Tensor

from jvp_flash_attention.jvp_attention import JVPAttn


def _parse_dtype(value: str) -> torch.dtype:
    """Convert a command-line dtype name to a supported torch dtype."""
    try:
        dtype = getattr(torch, value)
    except AttributeError as error:
        raise argparse.ArgumentTypeError(f"Unknown dtype {value!r}.") from error
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise argparse.ArgumentTypeError(f"Unsupported dtype {value!r}.")
    return dtype


def _time(function, *, warmup: int, repeats: int) -> list[float]:
    """Return per-call CUDA-event timings in milliseconds."""
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
    return samples


def main() -> None:
    """Run the dense fused-primal/JVP steady-state benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--sequence", type=int, default=3904)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    inputs = tuple(
        torch.randn(
            args.batch,
            args.heads,
            args.sequence,
            args.head_dim,
            dtype=args.dtype,
            device="cuda",
        )
        for _ in range(6)
    )

    def fused_dual(*values: Tensor) -> tuple[Tensor, Tensor]:
        def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return JVPAttn.fwd_dual(q, k, v, USE_TMA=True)

        return torch.func.jvp(attention, values[:3], values[3:])

    function = torch.compile(fused_dual, fullgraph=True) if args.compiled else fused_dual
    function(*inputs)
    torch.cuda.synchronize()
    samples = _time(lambda: function(*inputs), warmup=args.warmup, repeats=args.repeats)
    ordered = sorted(samples)
    payload = {
        "package_source": str(Path(inspect.getfile(JVPAttn)).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "dtype": str(args.dtype).removeprefix("torch."),
        "shape": [args.batch, args.heads, args.sequence, args.head_dim],
        "compiled": args.compiled,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "median_ms": statistics.median(ordered),
        "p20_ms": ordered[int(0.2 * (len(ordered) - 1))],
        "p80_ms": ordered[int(0.8 * (len(ordered) - 1))],
        "minimum_ms": ordered[0],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

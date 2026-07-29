from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch
from torch import Tensor

from jvp_flash_attention import BlockSparseMask, JVPAttn

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@dataclass(frozen=True)
class CoveringCase:
    dtype: torch.dtype
    head_dim: int
    length: int
    batch: int
    heads: int
    pattern: str
    tangent_source: str
    use_tma: bool


CASES = (
    CoveringCase(torch.float32, 16, 32, 1, 1, "full", "q", False),
    CoveringCase(torch.float16, 32, 60, 2, 1, "causal", "k", True),
    CoveringCase(torch.bfloat16, 64, 64, 1, 4, "block_diagonal", "v", False),
    CoveringCase(torch.float32, 128, 96, 2, 4, "sliding", "all", True),
    CoveringCase(torch.float16, 256, 127, 1, 1, "random_tiles", "q", False),
    CoveringCase(torch.bfloat16, 64, 128, 2, 1, "mixed", "k", True),
    CoveringCase(torch.float32, 32, 257, 1, 4, "mixed", "v", False),
)
TMA_CASES = tuple(replace(case, use_tma=use_tma) for case in CASES for use_tma in (False, True))


def _mask(case: CoveringCase) -> Tensor:
    length = case.length
    mask = torch.zeros(length, length, dtype=torch.bool, device="cuda")
    if case.pattern == "full":
        mask.fill_(True)
    elif case.pattern == "causal":
        mask.copy_(torch.ones_like(mask).tril())
    elif case.pattern == "block_diagonal":
        for start in range(0, length, 32):
            mask[start : start + 32, start : start + 32] = True
    elif case.pattern == "sliding":
        for query in range(length):
            mask[query, max(0, query - 47) : query + 1] = True
    elif case.pattern == "random_tiles":
        generator = torch.Generator(device="cuda").manual_seed(91)
        blocks = (length + 31) // 32
        occupied = torch.rand(blocks, blocks, generator=generator, device="cuda") > 0.7
        occupied.diagonal().fill_(True)
        mask = occupied.repeat_interleave(32, 0).repeat_interleave(32, 1)[:length, :length]
    elif case.pattern == "mixed":
        generator = torch.Generator(device="cuda").manual_seed(92)
        mask = (
            torch.rand(
                length,
                length,
                generator=generator,
                device="cuda",
            )
            > 0.82
        )
        mask.diagonal().fill_(True)
        mask[: min(32, length), : min(32, length)] = True
    else:
        raise ValueError(case.pattern)
    return mask


def _explicit_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
    scores *= q.shape[-1] ** -0.5
    return torch.matmul(scores.masked_fill(~mask, -torch.inf).softmax(-1), v.float())


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        # Triton tensor-core dots use TF32 for float32 inputs.
        return 2e-2, 2e-2
    if dtype == torch.float16:
        return 2e-2, 2e-2
    return 4e-2, 4e-2


@pytest.mark.parametrize(
    "case",
    TMA_CASES,
    ids=lambda case: (
        f"{str(case.dtype).removeprefix('torch.')}-d{case.head_dim}-n{case.length}-"
        f"b{case.batch}h{case.heads}-{case.pattern}-{case.tangent_source}-"
        f"tma{int(case.use_tma)}"
    ),
)
def test_sparse_covering_matrix(case: CoveringCase) -> None:
    torch.manual_seed(900 + case.length)
    mask = _mask(case)
    block_mask = BlockSparseMask.from_bool(mask)
    primals = tuple(
        (
            0.5
            * torch.randn(
                case.batch,
                case.heads,
                case.length,
                case.head_dim,
                dtype=case.dtype,
                device="cuda",
            )
        ).requires_grad_()
        for _ in range(3)
    )
    raw_tangents = tuple(torch.randn_like(tensor) for tensor in primals)
    tangent_index = {"q": 0, "k": 1, "v": 2}
    tangents = tuple(
        (
            tangent
            if case.tangent_source == "all" or tangent_index[case.tangent_source] == index
            else torch.zeros_like(tangent)
        )
        for index, tangent in enumerate(raw_tangents)
    )

    def sparse(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return JVPAttn.fwd_dual(
            q,
            k,
            v,
            block_mask=block_mask,
            USE_TMA=case.use_tma,
        )

    primal, tangent = torch.func.jvp(sparse, primals, tangents)
    cotangent = torch.randn_like(primal)
    actual_grads = torch.autograd.grad(primal, primals, cotangent)

    reference_primals = tuple(tensor.detach().float().requires_grad_() for tensor in primals)
    reference_tangents = tuple(tensor.float() for tensor in tangents)
    reference_primal, reference_tangent = torch.func.jvp(
        lambda q, k, v: _explicit_attention(q, k, v, mask),
        reference_primals,
        reference_tangents,
    )
    reference_grads = torch.autograd.grad(
        reference_primal,
        reference_primals,
        cotangent.float(),
    )

    atol, rtol = _tolerances(case.dtype)
    torch.testing.assert_close(primal.float(), reference_primal, atol=atol, rtol=rtol)
    torch.testing.assert_close(tangent.float(), reference_tangent, atol=atol, rtol=rtol)
    for actual, expected in zip(actual_grads, reference_grads, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)

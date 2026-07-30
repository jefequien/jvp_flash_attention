from __future__ import annotations

import pytest
import torch
from torch import Tensor

from jvp_flash_attention import (
    AttentionImplementation,
    JVPAttn,
    flash_attention,
)
from jvp_flash_attention.jvp_attention import supports_tma

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _inputs(
    head_dim: int = 64,
    query_length: int = 96,
    key_value_length: int = 64,
) -> tuple[Tensor, ...]:
    torch.manual_seed(47)
    q = torch.randn(
        1,
        2,
        query_length,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        1,
        2,
        key_value_length,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    return q, k, v, torch.randn_like(q), torch.randn_like(k), torch.randn_like(v)


def _explicit(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
) -> Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
    if causal:
        causal_mask = torch.ones(
            q.shape[-2],
            k.shape[-2],
            dtype=torch.bool,
            device=q.device,
        ).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    return torch.matmul((scores * (q.shape[-1] ** -0.5)).softmax(dim=-1), v.float())


@pytest.mark.parametrize("query_only", [False, True])
@pytest.mark.parametrize("head_dim", [16, 64])
@pytest.mark.parametrize(
    ("query_length", "key_value_length"),
    [(96, 64), (64, 96)],
)
def test_rectangular_jvp_and_backward_match_explicit(
    query_only: bool,
    head_dim: int,
    query_length: int,
    key_value_length: int,
) -> None:
    q, k, v, tangent_q, tangent_k, tangent_v = _inputs(
        head_dim,
        query_length,
        key_value_length,
    )
    primals = tuple(tensor.requires_grad_() for tensor in (q, k, v))

    if query_only:
        expected, expected_tangent = torch.func.jvp(
            lambda a: _explicit(a, k, v),
            (q,),
            (tangent_q,),
        )
        actual, actual_tangent = torch.func.jvp(
            lambda a: flash_attention(
                a,
                k,
                v,
                implementation=AttentionImplementation.DENSE_POINTER,
            ),
            (q,),
            (tangent_q,),
        )
    else:
        expected, expected_tangent = torch.func.jvp(
            _explicit,
            primals,
            (tangent_q, tangent_k, tangent_v),
        )
        actual, actual_tangent = torch.func.jvp(
            lambda a, b, c: flash_attention(
                a,
                b,
                c,
                implementation=AttentionImplementation.DENSE_POINTER,
            ),
            primals,
            (tangent_q, tangent_k, tangent_v),
        )

    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        actual_tangent.float(),
        expected_tangent,
        rtol=3e-2,
        atol=3e-2,
    )

    actual_grads = torch.autograd.grad(actual.float().square().mean(), primals)
    expected_grads = torch.autograd.grad(expected.square().mean(), primals)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(
            actual_grad.float(),
            expected_grad.float(),
            rtol=3e-2,
            atol=3e-4,
        )


@pytest.mark.parametrize("causal", [False, True])
def test_square_head_dim_16_tma_request_matches_pointer_fallback(
    causal: bool,
) -> None:
    if not supports_tma():
        pytest.skip("TMA fallback regression requires a TMA-capable GPU")
    q, k, v, tangent_q, tangent_k, tangent_v = _inputs(
        head_dim=16,
        query_length=64,
        key_value_length=64,
    )
    primals = tuple(tensor.requires_grad_() for tensor in (q, k, v))
    tangents = (tangent_q, tangent_k, tangent_v)

    expected, expected_tangent = torch.func.jvp(
        lambda a, b, c: _explicit(a, b, c, causal=causal),
        primals,
        tangents,
    )
    actual, actual_tangent = torch.func.jvp(
        lambda a, b, c: JVPAttn.fwd_dual(
            a,
            b,
            c,
            causal=causal,
            USE_TMA=True,
        ),
        primals,
        tangents,
    )

    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        actual_tangent.float(),
        expected_tangent,
        rtol=3e-2,
        atol=3e-2,
    )


def test_explicit_dense_tma_rejects_pointer_only_geometry() -> None:
    q, k, v, *_ = _inputs(
        head_dim=16,
        query_length=64,
        key_value_length=64,
    )
    with pytest.raises(ValueError, match="head dimension"):
        flash_attention(
            q,
            k,
            v,
            implementation=AttentionImplementation.DENSE_TMA,
        )

    q, k, v, *_ = _inputs(
        head_dim=64,
        query_length=96,
        key_value_length=64,
    )
    with pytest.raises(ValueError, match="square attention"):
        flash_attention(
            q,
            k,
            v,
            implementation=AttentionImplementation.DENSE_TMA,
        )


def test_compiled_rectangular_query_jvp_and_backward_match_eager() -> None:
    q, k, v, tangent_q, _, _ = _inputs()

    def fused(
        a: Tensor, b: Tensor, c: Tensor, tangent: Tensor
    ) -> tuple[Tensor, Tensor]:
        return torch.func.jvp(
            lambda query: flash_attention(
                query,
                b,
                c,
                implementation=AttentionImplementation.DENSE_POINTER,
            ),
            (a,),
            (tangent,),
        )

    def run(function, primals):
        primal, tangent = function(*primals, tangent_q)
        gradients = torch.autograd.grad(
            primal.float().square().mean(),
            primals,
        )
        return primal, tangent, gradients

    eager_primals = tuple(tensor.detach().requires_grad_() for tensor in (q, k, v))
    expected = run(fused, eager_primals)
    compiled_primals = tuple(tensor.detach().requires_grad_() for tensor in (q, k, v))
    actual = run(torch.compile(fused, fullgraph=True), compiled_primals)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    for actual_grad, expected_grad in zip(actual[2], expected[2], strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)

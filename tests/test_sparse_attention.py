from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.autograd.forward_ad as fw_ad
from torch import Tensor

from jvp_flash_attention import BlockSparseMask, JVPAttn
from tests.jit_sandbox_masks import make_pmf_mask

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _explicit_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, -torch.inf)
    return torch.matmul(scores.softmax(dim=-1), v.float())


def _mixed_mask(length: int, *, device: str = "cuda") -> Tensor:
    generator = torch.Generator(device=device).manual_seed(123)
    mask = torch.rand(length, length, generator=generator, device=device) > 0.78
    mask.diagonal().fill_(True)
    if length >= 32:
        mask[:32, :32] = True
    if length > 32:
        mask[:32, 32:] = False
    return mask


def _inputs(
    *,
    batch: int = 1,
    heads: int = 2,
    length: int = 53,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, ...]:
    torch.manual_seed(321)
    return tuple(
        torch.randn(
            batch,
            heads,
            length,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        for _ in range(6)
    )


@pytest.mark.parametrize("use_tma", [False, True])
def test_sparse_primal_jvp_and_backward(use_tma: bool) -> None:
    mask = _mixed_mask(53)
    block_mask = BlockSparseMask.from_bool(mask)
    q, k, v, tangent_q, tangent_k, tangent_v = _inputs()
    scale = 0.2

    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()

    def sparse(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        return JVPAttn.fwd_dual(
            a,
            b,
            c,
            block_mask=block_mask,
            sm_scale=scale,
            USE_TMA=use_tma,
        )

    primal, tangent = torch.func.jvp(
        sparse,
        (q, k, v),
        (tangent_q, tangent_k, tangent_v),
    )
    reference_primal, reference_tangent = torch.func.jvp(
        lambda a, b, c: _explicit_attention(a, b, c, mask, scale=scale),
        (q, k, v),
        (tangent_q, tangent_k, tangent_v),
    )
    pad_tokens = block_mask.padded_length - mask.shape[-1]
    dense_primals = tuple(
        torch.nn.functional.pad(tensor, (0, 0, 0, pad_tokens)) for tensor in (q, k, v)
    )
    dense_tangents = tuple(
        torch.nn.functional.pad(tensor, (0, 0, 0, pad_tokens))
        for tensor in (tangent_q, tangent_k, tangent_v)
    )
    dense_primal, dense_tangent = torch.func.jvp(
        lambda a, b, c: JVPAttn.fwd_dual(
            a,
            b,
            c,
            attn_mask=block_mask.to_dense(padded=True)[0, 0],
            sm_scale=scale,
            USE_TMA=False,
        )[..., : mask.shape[-1], :],
        dense_primals,
        dense_tangents,
    )
    torch.testing.assert_close(primal.float(), reference_primal, atol=1.5e-2, rtol=2e-2)
    torch.testing.assert_close(tangent.float(), reference_tangent, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(primal, dense_primal, atol=1.5e-2, rtol=2e-2)
    torch.testing.assert_close(tangent, dense_tangent, atol=2e-2, rtol=2e-2)

    output_cotangent = torch.randn_like(primal)
    (primal * output_cotangent).sum().backward()
    actual_grads = (q.grad.float(), k.grad.float(), v.grad.float())

    reference_inputs = tuple(tensor.detach().float().requires_grad_() for tensor in (q, k, v))
    reference_output = _explicit_attention(
        *reference_inputs,
        mask,
        scale=scale,
    )
    reference_grads = torch.autograd.grad(
        reference_output,
        reference_inputs,
        output_cotangent.float(),
    )
    for actual, expected in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=1.5e-2, rtol=2e-2)


def test_direct_dual_matches_torch_func_jvp() -> None:
    mask = _mixed_mask(53)
    block_mask = BlockSparseMask.from_bool(mask)
    q, k, v, tangent_q, tangent_k, tangent_v = _inputs()

    def attention(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        return JVPAttn.fwd_dual(a, b, c, block_mask=block_mask)

    expected_primal, expected_tangent = torch.func.jvp(
        attention,
        (q, k, v),
        (tangent_q, tangent_k, tangent_v),
    )
    with fw_ad.dual_level():
        duals = tuple(
            fw_ad.make_dual(primal, tangent)
            for primal, tangent in zip(
                (q, k, v),
                (tangent_q, tangent_k, tangent_v),
                strict=True,
            )
        )
        dual_output = attention(*duals)
        actual_primal, actual_tangent = fw_ad.unpack_dual(dual_output)

    torch.testing.assert_close(actual_primal, expected_primal)
    torch.testing.assert_close(actual_tangent, expected_tangent)


@pytest.mark.parametrize("sparse", [False, True])
def test_direct_dual_supports_a_single_input_tangent(sparse: bool) -> None:
    mask = _mixed_mask(64)
    block_mask = BlockSparseMask.from_bool(mask)
    q, k, v, _, tangent_k, _ = _inputs(length=64)

    def attention(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        mask_argument = {"block_mask": block_mask} if sparse else {"attn_mask": mask}
        return JVPAttn.fwd_dual(a, b, c, sm_scale=0.2, **mask_argument)

    expected_primal, expected_tangent = torch.func.jvp(
        attention,
        (q, k, v),
        (torch.zeros_like(q), tangent_k, torch.zeros_like(v)),
    )
    with fw_ad.dual_level():
        dual_k = fw_ad.make_dual(k, tangent_k)
        dual_output = attention(q, dual_k, v)
        actual_primal, actual_tangent = fw_ad.unpack_dual(dual_output)

    torch.testing.assert_close(actual_primal, expected_primal)
    assert actual_tangent is not None
    torch.testing.assert_close(actual_tangent, expected_tangent)


@pytest.mark.parametrize("use_tma", [False, True])
def test_compiled_sparse_jvp_uses_runtime_mask(use_tma: bool) -> None:
    mask = _mixed_mask(53)
    alternate_mask = mask.clone()
    alternate_mask[40, 33] = ~alternate_mask[40, 33]
    block_masks = (
        BlockSparseMask.from_bool(mask),
        BlockSparseMask.from_bool(alternate_mask),
    )
    inputs = _inputs()

    def fused_dual(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        tangent_q: Tensor,
        tangent_k: Tensor,
        tangent_v: Tensor,
        runtime_mask: BlockSparseMask,
    ) -> tuple[Tensor, Tensor]:
        def attention(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
            return JVPAttn.fwd_dual(
                a,
                b,
                c,
                block_mask=runtime_mask,
                sm_scale=0.2,
                USE_TMA=use_tma,
            )

        return torch.func.jvp(
            attention,
            (q, k, v),
            (tangent_q, tangent_k, tangent_v),
        )

    expected_outputs = tuple(fused_dual(*inputs, block_mask) for block_mask in block_masks)
    assert not torch.equal(expected_outputs[0][0], expected_outputs[1][0])
    compiled = torch.compile(fused_dual, fullgraph=True)
    actual_outputs = tuple(compiled(*inputs, block_mask) for block_mask in block_masks)
    for actual, expected in zip(actual_outputs, expected_outputs, strict=True):
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor)


def test_compiled_sparse_detached_tangent_backward_matches_eager() -> None:
    mask = _mixed_mask(53)
    block_mask = BlockSparseMask.from_bool(mask)
    inputs = _inputs(heads=1)

    def fused_dual(*args: Tensor) -> tuple[Tensor, Tensor]:
        def attention(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
            return JVPAttn.fwd_dual(
                a,
                b,
                c,
                block_mask=block_mask,
                USE_TMA=True,
            )

        return torch.func.jvp(attention, args[:3], args[3:])

    def run(
        function,
        primals: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        primal, tangent = function(*primals, *inputs[3:])
        compound = primal.float() + 0.2 * tangent.detach().float()
        loss = compound.square().mean()
        loss.backward()
        return loss.detach(), tuple(tensor.grad.detach().clone() for tensor in primals)

    eager_primals = tuple(tensor.detach().requires_grad_() for tensor in inputs[:3])
    expected_loss, expected_grads = run(fused_dual, eager_primals)
    compiled_primals = tuple(tensor.detach().requires_grad_() for tensor in inputs[:3])
    compiled = torch.compile(fused_dual, fullgraph=True)
    actual_loss, actual_grads = run(compiled, compiled_primals)

    torch.testing.assert_close(actual_loss, expected_loss)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)


def test_compiled_dense_jvp_backward_regression() -> None:
    mask = _mixed_mask(64)
    inputs = _inputs(heads=1, length=64)

    def fused_dual(*args: Tensor) -> tuple[Tensor, Tensor]:
        def attention(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
            return JVPAttn.fwd_dual(
                a,
                b,
                c,
                attn_mask=mask,
                USE_TMA=False,
            )

        return torch.func.jvp(attention, args[:3], args[3:])

    eager_primals = tuple(tensor.detach().requires_grad_() for tensor in inputs[:3])
    expected_primal, expected_tangent = fused_dual(*eager_primals, *inputs[3:])
    expected_grads = torch.autograd.grad(
        expected_primal,
        eager_primals,
        torch.ones_like(expected_primal),
    )

    compiled_primals = tuple(tensor.detach().requires_grad_() for tensor in inputs[:3])
    compiled = torch.compile(fused_dual, fullgraph=True)
    actual_primal, actual_tangent = compiled(*compiled_primals, *inputs[3:])
    assert not actual_tangent.requires_grad
    actual_grads = torch.autograd.grad(
        actual_primal,
        compiled_primals,
        torch.ones_like(actual_primal),
    )

    torch.testing.assert_close(actual_primal, expected_primal)
    torch.testing.assert_close(actual_tangent, expected_tangent)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["dense", "causal", "sparse"])
def test_compiled_primal_backward_matches_eager(kind: str) -> None:
    mask = _mixed_mask(64)
    block_mask = BlockSparseMask.from_bool(mask)
    inputs = _inputs(heads=1, length=64)[:3]
    cotangent = torch.randn_like(inputs[0])

    def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if kind == "sparse":
            return JVPAttn.fwd(q, k, v, block_mask=block_mask, USE_TMA=True)
        if kind == "causal":
            return JVPAttn.fwd(q, k, v, causal=True, USE_TMA=False)
        return JVPAttn.fwd(q, k, v, attn_mask=mask, USE_TMA=False)

    def run(function, primals):
        output = function(*primals)
        gradients = torch.autograd.grad(output, primals, cotangent)
        return output, gradients

    eager_primals = tuple(tensor.detach().requires_grad_() for tensor in inputs)
    expected_output, expected_gradients = run(attention, eager_primals)
    compiled_primals = tuple(tensor.detach().requires_grad_() for tensor in inputs)
    actual_output, actual_gradients = run(
        torch.compile(attention, fullgraph=True),
        compiled_primals,
    )

    torch.testing.assert_close(actual_output, expected_output)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("mask_batch", "mask_heads"),
    [(1, 1), (2, 1), (1, 4), (2, 4)],
)
@pytest.mark.parametrize("use_tma", [False, True])
def test_sparse_mask_batch_head_broadcast(
    mask_batch: int,
    mask_heads: int,
    use_tma: bool,
) -> None:
    batch, heads, length, head_dim = 2, 4, 64, 32
    masks = torch.empty(mask_batch, mask_heads, length, length, dtype=torch.bool, device="cuda")
    for batch_idx in range(mask_batch):
        for head_idx in range(mask_heads):
            masks[batch_idx, head_idx] = _mixed_mask(length)
            visible = 8 * (1 + (batch_idx + head_idx) % 4)
            masks[batch_idx, head_idx, :, :visible] = True
    block_mask = BlockSparseMask.from_bool(masks)
    q, k, v, *_ = _inputs(
        batch=batch,
        heads=heads,
        length=length,
        head_dim=head_dim,
    )
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()

    actual = JVPAttn.fwd(q, k, v, block_mask=block_mask, USE_TMA=use_tma)
    expected = _explicit_attention(q, k, v, masks)
    torch.testing.assert_close(actual.float(), expected, atol=1.5e-2, rtol=2e-2)

    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (q, k, v), cotangent)
    reference_inputs = tuple(tensor.detach().float().requires_grad_() for tensor in (q, k, v))
    reference_output = _explicit_attention(*reference_inputs, masks)
    reference_gradients = torch.autograd.grad(
        reference_output,
        reference_inputs,
        cotangent.float(),
    )
    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient.float(),
            reference_gradient,
            atol=1.5e-2,
            rtol=2e-2,
        )


def test_dense_mask_broadcast_regression() -> None:
    q, k, v, *_ = _inputs(length=64)
    mask = _mixed_mask(64)
    expected = _explicit_attention(q, k, v, mask)

    for dense_mask in (mask, mask[None, None]):
        actual = JVPAttn.fwd(
            q,
            k,
            v,
            attn_mask=dense_mask,
            USE_TMA=False,
        )
        torch.testing.assert_close(actual.float(), expected, atol=1.5e-2, rtol=2e-2)


def test_cpu_constructed_block_mask_can_move_to_cuda() -> None:
    cpu_mask = _mixed_mask(53, device="cpu")
    block_mask = BlockSparseMask.from_bool(cpu_mask).to("cuda")
    q, k, v, *_ = _inputs(heads=1)

    actual = JVPAttn.fwd(q, k, v, block_mask=block_mask, USE_TMA=False)
    expected = _explicit_attention(q, k, v, cpu_mask.to("cuda"))
    torch.testing.assert_close(actual.float(), expected, atol=1.5e-2, rtol=2e-2)


def test_dense_no_mask_causal_additive_and_dropout_regressions() -> None:
    q, k, v, *_ = _inputs(length=64)
    no_mask_expected = _explicit_attention(
        q,
        k,
        v,
        torch.ones(64, 64, dtype=torch.bool, device="cuda"),
    )
    no_mask_actual = JVPAttn.fwd(q, k, v, USE_TMA=False)
    torch.testing.assert_close(no_mask_actual.float(), no_mask_expected, atol=1.5e-2, rtol=2e-2)

    causal_mask = torch.ones(64, 64, dtype=torch.bool, device="cuda").tril()
    causal_expected = _explicit_attention(q, k, v, causal_mask)
    causal_actual = JVPAttn.fwd(q, k, v, causal=True, USE_TMA=False)
    torch.testing.assert_close(causal_actual.float(), causal_expected, atol=1.5e-2, rtol=2e-2)

    boolean_mask = _mixed_mask(64)
    additive_mask = torch.where(
        boolean_mask,
        torch.zeros((), dtype=q.dtype, device=q.device),
        torch.full((), -1e2, dtype=q.dtype, device=q.device),
    )
    boolean_actual = JVPAttn.fwd(q, k, v, attn_mask=boolean_mask, USE_TMA=False)
    additive_actual = JVPAttn.fwd(q, k, v, attn_mask=additive_mask, USE_TMA=False)
    torch.testing.assert_close(additive_actual, boolean_actual)

    with pytest.raises(NotImplementedError, match="Dropout"):
        JVPAttn.fwd(q, k, v, dropout_p=0.1)


def test_pmf_style_detached_tangent_training() -> None:
    fixture = make_pmf_mask(
        grid_shape=(5, 4, 4),
        target_start_token=16,
        device="cuda",
    )
    block_mask = BlockSparseMask.from_bool(fixture.mask)
    q, k, v, tangent_q, tangent_k, tangent_v = _inputs(
        length=fixture.sequence_length,
        head_dim=32,
    )
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()

    def sparse(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        return JVPAttn.fwd_dual(a, b, c, block_mask=block_mask)

    primal, tangent = torch.func.jvp(
        sparse,
        (q, k, v),
        (tangent_q, tangent_k, tangent_v),
    )
    compound = primal.float() + 0.3 * tangent.detach().float()
    loss = compound.square().mean()
    loss.backward()
    actual_grads = (q.grad.float(), k.grad.float(), v.grad.float())

    reference_inputs = tuple(tensor.detach().float().requires_grad_() for tensor in (q, k, v))
    reference_primal, reference_tangent = torch.func.jvp(
        lambda a, b, c: _explicit_attention(a, b, c, fixture.mask),
        reference_inputs,
        tuple(t.float() for t in (tangent_q, tangent_k, tangent_v)),
    )
    reference_compound = reference_primal + 0.3 * reference_tangent.detach()
    reference_loss = reference_compound.square().mean()
    reference_grads = torch.autograd.grad(reference_loss, reference_inputs)

    torch.testing.assert_close(loss, reference_loss, atol=2e-4, rtol=2e-3)
    for actual, expected in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


def test_float32_jvp_matches_central_difference() -> None:
    length, head_dim = 32, 32
    mask = _mixed_mask(length)
    block_mask = BlockSparseMask.from_bool(mask)
    inputs = _inputs(length=length, head_dim=head_dim, dtype=torch.float32)
    primals, tangents = inputs[:3], inputs[3:]

    def attention(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        return JVPAttn.fwd_dual(
            a,
            b,
            c,
            block_mask=block_mask,
            USE_TMA=False,
        )

    _, tangent = torch.func.jvp(attention, primals, tangents)
    # Triton's float32 tensor-core dot uses TF32 on NVIDIA. A moderately sized
    # step avoids measuring TF32 quantization noise instead of the derivative.
    epsilon = 2e-2
    plus = attention(
        *(primal + epsilon * direction for primal, direction in zip(primals, tangents))
    )
    minus = attention(
        *(primal - epsilon * direction for primal, direction in zip(primals, tangents))
    )
    finite_difference = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(tangent, finite_difference, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("use_tma", [False, True])
def test_masked_logits_do_not_affect_online_softmax_max(use_tma: bool) -> None:
    length, head_dim = 64, 16
    mask = torch.zeros(length, length, dtype=torch.bool, device="cuda")
    mask[0, 0] = True
    mask[1:, 32] = True
    block_mask = BlockSparseMask.from_bool(mask)

    q = torch.zeros(1, 1, length, head_dim, device="cuda")
    k = torch.zeros_like(q)
    v = torch.randn_like(q)
    q[..., 0] = 100
    k[:, :, 0, 0] = -100
    k[:, :, 32, 0] = -100
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    tangents = (torch.zeros_like(q), torch.zeros_like(k), torch.randn_like(v))

    def attention(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        return JVPAttn.fwd_dual(
            a,
            b,
            c,
            block_mask=block_mask,
            USE_TMA=use_tma,
        )

    primal, tangent = torch.func.jvp(attention, (q, k, v), tangents)
    visible_keys = torch.tensor([0, *([32] * 63)], device="cuda")
    expected_primal = v[:, :, visible_keys]
    expected_tangent = tangents[2][:, :, visible_keys]
    assert torch.isfinite(primal).all()
    assert torch.isfinite(tangent).all()
    torch.testing.assert_close(primal, expected_primal, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(tangent, expected_tangent, atol=5e-3, rtol=5e-3)

    cotangent = torch.randn_like(primal)
    actual_gradients = torch.autograd.grad(primal, (q, k, v), cotangent)
    assert all(torch.isfinite(gradient).all() for gradient in actual_gradients)


def test_dense_and_sparse_masks_are_mutually_exclusive() -> None:
    q, k, v, *_ = _inputs(length=32, head_dim=32)
    mask = torch.eye(32, dtype=torch.bool, device="cuda")
    block_mask = BlockSparseMask.from_bool(mask)
    with pytest.raises(ValueError, match="mutually exclusive"):
        JVPAttn.fwd(q, k, v, attn_mask=mask, block_mask=block_mask)


def test_dense_mask_geometry_is_checked_when_value_checks_are_disabled() -> None:
    q, k, v, *_ = _inputs(length=32, head_dim=32)
    malformed = torch.ones(32, 31, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="must have shape"):
        JVPAttn.fwd(
            q,
            k,
            v,
            attn_mask=malformed,
            verify_attn_mask=False,
        )
    with pytest.raises(ValueError, match="matching Q/K/V shapes"):
        JVPAttn.fwd(q, k[..., :-1, :], v)
    with pytest.raises(TypeError, match="matching Q/K/V dtypes"):
        JVPAttn.fwd(q, k.float(), v)
    with pytest.raises(ValueError, match="same device"):
        JVPAttn.fwd(
            q,
            k,
            v,
            attn_mask=torch.ones(32, 32, dtype=torch.bool),
            verify_attn_mask=False,
        )


def test_sparse_rejects_unsupported_dtype() -> None:
    block_mask = BlockSparseMask.from_bool(torch.eye(32, dtype=torch.bool, device="cuda"))
    q, k, v = (torch.randn(1, 1, 32, 16, dtype=torch.float64, device="cuda") for _ in range(3))
    with pytest.raises(TypeError, match="supports float16"):
        JVPAttn.fwd(q, k, v, block_mask=block_mask)


def test_sparse_rejects_noncanonical_mask_metadata() -> None:
    q, k, v, *_ = _inputs(length=32, head_dim=32)
    block_mask = BlockSparseMask.from_bool(torch.eye(32, dtype=torch.bool, device="cuda"))
    with pytest.raises(ValueError, match="block_size"):
        JVPAttn.fwd(q, k, v, block_mask=replace(block_mask, block_size=16))
    with pytest.raises(ValueError, match="padded_length"):
        JVPAttn.fwd(q, k, v, block_mask=replace(block_mask, padded_length=64))

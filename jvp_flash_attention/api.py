"""Public, explicit dispatch API for JVP Flash Attention."""

from __future__ import annotations

from enum import Enum

from torch import Tensor
from torch.autograd import forward_ad as fwAD

from .block_sparse import BlockSparseMask
from .jvp_attention import JVPAttn, _is_compiling, supports_tma
from .sparse_attention import supports_sparse_tma


class AttentionImplementation(str, Enum):
    """Concrete attention implementation requested by the caller."""

    DENSE_POINTER = "dense_pointer"
    DENSE_TMA = "dense_tma"
    BLOCK_SPARSE_POINTER = "block_sparse_pointer"
    BLOCK_SPARSE_TMA = "block_sparse_tma"

    @property
    def is_block_sparse(self) -> bool:
        """Whether this implementation consumes a ``BlockSparseMask``."""
        return self in {
            AttentionImplementation.BLOCK_SPARSE_POINTER,
            AttentionImplementation.BLOCK_SPARSE_TMA,
        }

    @property
    def uses_tma(self) -> bool:
        """Whether this implementation uses tensor memory access descriptors."""
        return self in {
            AttentionImplementation.DENSE_TMA,
            AttentionImplementation.BLOCK_SPARSE_TMA,
        }


def _coerce_implementation(
    implementation: AttentionImplementation | str,
) -> AttentionImplementation:
    if isinstance(implementation, AttentionImplementation):
        return implementation
    try:
        return AttentionImplementation(implementation)
    except ValueError as error:
        choices = ", ".join(item.value for item in AttentionImplementation)
        raise ValueError(f"Unknown attention implementation {implementation!r}; expected one of {choices}.") from error


def _has_forward_ad_tangent(*tensors: Tensor) -> bool:
    return any(fwAD.unpack_dual(tensor).tangent is not None for tensor in tensors)


def _validate_implementation(
    q: Tensor,
    k: Tensor,
    implementation: AttentionImplementation,
    attn_mask: Tensor | None,
    block_mask: BlockSparseMask | None,
    causal: bool,
) -> None:
    if implementation.is_block_sparse:
        if not isinstance(block_mask, BlockSparseMask):
            raise TypeError(f"{implementation.value} requires a BlockSparseMask, got {type(block_mask).__name__}.")
        if attn_mask is not None:
            raise ValueError(f"{implementation.value} does not accept attn_mask.")
        if causal:
            raise ValueError(f"{implementation.value} expresses causality through block_mask, not causal=True.")
    elif block_mask is not None:
        raise ValueError(f"{implementation.value} does not accept block_mask.")
    if q.device.type != "cuda":
        raise RuntimeError(f"{implementation.value} requires a CUDA or ROCm device, got {q.device}.")
    if implementation is AttentionImplementation.DENSE_TMA:
        if q.ndim == 4 and k.ndim == 4 and q.shape[2] != k.shape[2]:
            raise ValueError("dense_tma supports only square attention.")
        if q.ndim == 4 and q.shape[-1] < 32:
            raise ValueError("dense_tma requires a head dimension of at least 32.")
        if _is_compiling():
            raise RuntimeError("dense_tma is not supported under torch.compile; use dense_pointer.")
        if not supports_tma():
            raise RuntimeError("dense_tma is unavailable on the current device.")
    elif (
        implementation is AttentionImplementation.BLOCK_SPARSE_TMA
        and q.ndim == 4
        and k.ndim == 4
        and q.shape[2] != k.shape[2]
    ):
        raise ValueError(
            "block_sparse_tma supports only square attention; use block_sparse_pointer for rectangular attention."
        )
    elif implementation is AttentionImplementation.BLOCK_SPARSE_TMA and not supports_sparse_tma(q.device):
        raise RuntimeError(f"block_sparse_tma is unavailable on {q.device}.")


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    implementation: AttentionImplementation | str,
    attn_mask: Tensor | None = None,
    block_mask: BlockSparseMask | None = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    sm_scale: float | None = None,
    warp_specialize: bool = True,
    verify_attn_mask: bool = True,
) -> Tensor:
    """Run one explicitly selected primal/JVP attention implementation.

    ``implementation`` is strict: this function never substitutes another
    kernel. It accepts primal tensors directly and preserves forward-AD
    tangents when called inside ``torch.func.jvp`` or a dual level.
    """
    selected = _coerce_implementation(implementation)
    _validate_implementation(q, k, selected, attn_mask, block_mask, causal)
    attention_fn = JVPAttn.fwd_dual if _has_forward_ad_tangent(q, k, v) else JVPAttn.fwd
    return attention_fn(
        q,
        k,
        v,
        attn_mask=attn_mask,
        block_mask=block_mask,
        dropout_p=dropout_p,
        causal=causal,
        sm_scale=sm_scale,
        warp_specialize=warp_specialize,
        USE_TMA=selected.uses_tma,
        verify_attn_mask=verify_attn_mask,
    )


__all__ = ["AttentionImplementation", "flash_attention"]

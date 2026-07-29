"""Triton kernels for block-sparse primal, JVP, and reverse attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from .block_sparse import BlockSparseMask

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ModuleNotFoundError:  # pragma: no cover - depends on the Triton build.
    TensorDescriptor = None

BLOCK_SIZE = 32
_TRITON_BLOCK_SIZE = tl.constexpr(BLOCK_SIZE)


@triton.jit
def _sparse_fwd_tile(
    acc,
    tangent_acc,
    l_i,
    m_i,
    tangent_mass,
    tangent_value_acc,
    q,
    tangent_q,
    K,
    V,
    TangentK,
    TangentV,
    PartialMasks,
    k_offset,
    v_offset,
    tangent_k_offset,
    tangent_v_offset,
    kv_block,
    mask_id,
    sm_scale,
    qk_scale,
    stride_kn,
    stride_kk,
    stride_vk,
    stride_vn,
    stride_tkn,
    stride_tkk,
    stride_tvk,
    stride_tvn,
    HEAD_DIM: tl.constexpr,
    ENABLE_JVP: tl.constexpr,
    IS_PARTIAL: tl.constexpr,
):
    """Accumulate one scheduled KV tile into an online-softmax Q tile."""
    offs_n = kv_block * _TRITON_BLOCK_SIZE + tl.arange(0, _TRITON_BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    k = tl.load(K + k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn)
    v = tl.load(V + v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn)

    qk = tl.dot(q, k)
    if ENABLE_JVP:
        tangent_k = tl.load(
            TangentK
            + tangent_k_offset
            + offs_d[:, None] * stride_tkk
            + offs_n[None, :] * stride_tkn
        )
        tangent_qk = tl.dot(tangent_q, k) + tl.dot(q, tangent_k)

    scaled_qk = qk * qk_scale
    if IS_PARTIAL:
        mask_offsets = (
            mask_id.to(tl.int64) * _TRITON_BLOCK_SIZE * _TRITON_BLOCK_SIZE
            + tl.arange(0, _TRITON_BLOCK_SIZE)[:, None] * _TRITON_BLOCK_SIZE
            + tl.arange(0, _TRITON_BLOCK_SIZE)[None, :]
        )
        element_mask = tl.load(PartialMasks + mask_offsets)
        scaled_qk = tl.where(element_mask, scaled_qk, -float("inf"))
        if ENABLE_JVP:
            tangent_qk = tl.where(element_mask, tangent_qk, 0.0)

    m_ij = tl.maximum(m_i, tl.max(scaled_qk, axis=1))
    if IS_PARTIAL:
        row_has_values = tl.sum(element_mask.to(tl.int32), axis=1) > 0
        m_ij = tl.where(row_has_values, m_ij, m_i)
    shifted_qk = scaled_qk - m_ij[:, None]
    if IS_PARTIAL:
        shifted_qk = tl.where(element_mask, shifted_qk, -float("inf"))
    alpha = tl.math.exp2(m_i - m_ij)
    probabilities = tl.math.exp2(shifted_qk)
    if IS_PARTIAL:
        alpha = tl.where(row_has_values, alpha, 1.0)

    l_i = l_i * alpha + tl.sum(probabilities, axis=1)
    acc *= alpha[:, None]

    probabilities_for_dot = probabilities.to(tl.float32)
    if ENABLE_JVP:
        tangent_probabilities = probabilities * (tangent_qk * sm_scale)
        tangent_acc *= alpha[:, None]
        tangent_acc = tl.dot(tangent_probabilities.to(tl.float32), v.to(tl.float32), tangent_acc)
        tangent_mass = tangent_mass * alpha + tl.sum(tangent_probabilities, axis=1)
        tangent_v = tl.load(
            TangentV
            + tangent_v_offset
            + offs_n[:, None] * stride_tvk
            + offs_d[None, :] * stride_tvn
        )
        tangent_value_acc = tangent_value_acc * alpha[:, None] + tl.dot(
            probabilities_for_dot, tangent_v.to(tl.float32)
        )

    acc = tl.dot(probabilities_for_dot, v.to(tl.float32), acc)
    return acc, tangent_acc, l_i, m_ij, tangent_mass, tangent_value_acc


@triton.jit
def _sparse_fwd_tile_tma(
    acc,
    tangent_acc,
    l_i,
    m_i,
    tangent_mass,
    tangent_value_acc,
    q,
    tangent_q,
    desc_k,
    desc_v,
    desc_tangent_k,
    desc_tangent_v,
    PartialMasks,
    sequence_offset,
    kv_block,
    mask_id,
    sm_scale,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    ENABLE_JVP: tl.constexpr,
    IS_PARTIAL: tl.constexpr,
):
    """TMA variant of one scheduled online-softmax tile update."""
    kv_offset = sequence_offset + kv_block * _TRITON_BLOCK_SIZE
    k = desc_k.load([kv_offset, 0]).T
    v = desc_v.load([kv_offset, 0])
    qk = tl.dot(q, k)
    if ENABLE_JVP:
        tangent_k = desc_tangent_k.load([kv_offset, 0]).T
        tangent_qk = tl.dot(tangent_q, k) + tl.dot(q, tangent_k)

    scaled_qk = qk * qk_scale
    if IS_PARTIAL:
        mask_offsets = (
            mask_id.to(tl.int64) * _TRITON_BLOCK_SIZE * _TRITON_BLOCK_SIZE
            + tl.arange(0, _TRITON_BLOCK_SIZE)[:, None] * _TRITON_BLOCK_SIZE
            + tl.arange(0, _TRITON_BLOCK_SIZE)[None, :]
        )
        element_mask = tl.load(PartialMasks + mask_offsets)
        scaled_qk = tl.where(element_mask, scaled_qk, -float("inf"))
        if ENABLE_JVP:
            tangent_qk = tl.where(element_mask, tangent_qk, 0.0)

    m_ij = tl.maximum(m_i, tl.max(scaled_qk, axis=1))
    if IS_PARTIAL:
        row_has_values = tl.sum(element_mask.to(tl.int32), axis=1) > 0
        m_ij = tl.where(row_has_values, m_ij, m_i)
    shifted_qk = scaled_qk - m_ij[:, None]
    if IS_PARTIAL:
        shifted_qk = tl.where(element_mask, shifted_qk, -float("inf"))
    alpha = tl.math.exp2(m_i - m_ij)
    probabilities = tl.math.exp2(shifted_qk)
    if IS_PARTIAL:
        alpha = tl.where(row_has_values, alpha, 1.0)
    l_i = l_i * alpha + tl.sum(probabilities, axis=1)
    acc *= alpha[:, None]
    probabilities_for_dot = probabilities.to(tl.float32)

    if ENABLE_JVP:
        tangent_probabilities = probabilities * (tangent_qk * sm_scale)
        tangent_acc *= alpha[:, None]
        tangent_acc = tl.dot(tangent_probabilities.to(tl.float32), v.to(tl.float32), tangent_acc)
        tangent_mass = tangent_mass * alpha + tl.sum(tangent_probabilities, axis=1)
        tangent_v = desc_tangent_v.load([kv_offset, 0])
        tangent_value_acc = tangent_value_acc * alpha[:, None] + tl.dot(
            probabilities_for_dot, tangent_v.to(tl.float32)
        )

    acc = tl.dot(probabilities_for_dot, v.to(tl.float32), acc)
    return acc, tangent_acc, l_i, m_ij, tangent_mass, tangent_value_acc


@triton.jit
def _attn_fwd_sparse(
    Q,
    K,
    V,
    TangentQ,
    TangentK,
    TangentV,
    Out,
    TangentOut,
    M,
    PartialKVNumBlocks,
    PartialKVIndices,
    PartialKVMaskIds,
    FullKVNumBlocks,
    FullKVIndices,
    PartialMasks,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_tqz,
    stride_tqh,
    stride_tqm,
    stride_tqd,
    stride_tkz,
    stride_tkh,
    stride_tkn,
    stride_tkk,
    stride_tvz,
    stride_tvh,
    stride_tvk,
    stride_tvn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    stride_toz,
    stride_toh,
    stride_tom,
    stride_tod,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MASK_BATCH: tl.constexpr,
    MASK_HEADS: tl.constexpr,
    MAX_PARTIAL: tl.constexpr,
    MAX_FULL: tl.constexpr,
    ENABLE_JVP: tl.constexpr,
):
    """Block-sparse fused primal/JVP attention forward."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // H
    head_idx = batch_head % H
    mask_batch_idx = batch_idx if MASK_BATCH > 1 else 0
    mask_head_idx = head_idx if MASK_HEADS > 1 else 0

    q_offset = batch_idx.to(tl.int64) * stride_qz + head_idx.to(tl.int64) * stride_qh
    k_offset = batch_idx.to(tl.int64) * stride_kz + head_idx.to(tl.int64) * stride_kh
    v_offset = batch_idx.to(tl.int64) * stride_vz + head_idx.to(tl.int64) * stride_vh
    tangent_q_offset = batch_idx.to(tl.int64) * stride_tqz + head_idx.to(tl.int64) * stride_tqh
    tangent_k_offset = batch_idx.to(tl.int64) * stride_tkz + head_idx.to(tl.int64) * stride_tkh
    tangent_v_offset = batch_idx.to(tl.int64) * stride_tvz + head_idx.to(tl.int64) * stride_tvh

    offs_m = query_block * _TRITON_BLOCK_SIZE + tl.arange(0, _TRITON_BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)

    acc = tl.zeros((_TRITON_BLOCK_SIZE, HEAD_DIM), tl.float32)
    l_i = tl.full((_TRITON_BLOCK_SIZE,), 1.0, tl.float32)
    m_i = tl.full((_TRITON_BLOCK_SIZE,), -float("inf"), tl.float32)

    if ENABLE_JVP:
        tangent_q = tl.load(
            TangentQ
            + tangent_q_offset
            + offs_m[:, None] * stride_tqm
            + offs_d[None, :] * stride_tqd
        )
        tangent_acc = tl.zeros((_TRITON_BLOCK_SIZE, HEAD_DIM), tl.float32)
        tangent_mass = tl.zeros((_TRITON_BLOCK_SIZE,), tl.float32)
        tangent_value_acc = tl.zeros((_TRITON_BLOCK_SIZE, HEAD_DIM), tl.float32)
    else:
        tangent_q = q
        tangent_acc = tl.zeros((1, 1), tl.float32)
        tangent_mass = tl.zeros((1,), tl.float32)
        tangent_value_acc = tl.zeros((1, 1), tl.float32)

    metadata_row = (mask_batch_idx * MASK_HEADS + mask_head_idx) * N_BLOCKS + query_block
    qk_scale = sm_scale * 1.4426950408889634

    full_count = tl.load(FullKVNumBlocks + metadata_row)
    full_row_offset = metadata_row.to(tl.int64) * MAX_FULL
    for entry_idx in range(full_count):
        kv_block = tl.load(FullKVIndices + full_row_offset + entry_idx)
        (
            acc,
            tangent_acc,
            l_i,
            m_i,
            tangent_mass,
            tangent_value_acc,
        ) = _sparse_fwd_tile(
            acc,
            tangent_acc,
            l_i,
            m_i,
            tangent_mass,
            tangent_value_acc,
            q,
            tangent_q,
            K,
            V,
            TangentK,
            TangentV,
            PartialMasks,
            k_offset,
            v_offset,
            tangent_k_offset,
            tangent_v_offset,
            kv_block,
            0,
            sm_scale,
            qk_scale,
            stride_kn,
            stride_kk,
            stride_vk,
            stride_vn,
            stride_tkn,
            stride_tkk,
            stride_tvk,
            stride_tvn,
            HEAD_DIM,
            ENABLE_JVP,
            IS_PARTIAL=False,
        )

    partial_count = tl.load(PartialKVNumBlocks + metadata_row)
    partial_row_offset = metadata_row.to(tl.int64) * MAX_PARTIAL
    for entry_idx in range(partial_count):
        kv_block = tl.load(PartialKVIndices + partial_row_offset + entry_idx)
        mask_id = tl.load(PartialKVMaskIds + partial_row_offset + entry_idx)
        (
            acc,
            tangent_acc,
            l_i,
            m_i,
            tangent_mass,
            tangent_value_acc,
        ) = _sparse_fwd_tile(
            acc,
            tangent_acc,
            l_i,
            m_i,
            tangent_mass,
            tangent_value_acc,
            q,
            tangent_q,
            K,
            V,
            TangentK,
            TangentV,
            PartialMasks,
            k_offset,
            v_offset,
            tangent_k_offset,
            tangent_v_offset,
            kv_block,
            mask_id,
            sm_scale,
            qk_scale,
            stride_kn,
            stride_kk,
            stride_vk,
            stride_vn,
            stride_tkn,
            stride_tkk,
            stride_tvk,
            stride_tvn,
            HEAD_DIM,
            ENABLE_JVP,
            IS_PARTIAL=True,
        )

    m_i += tl.math.log2(l_i)
    acc /= l_i[:, None]
    output_offset = batch_idx.to(tl.int64) * stride_oz + head_idx.to(tl.int64) * stride_oh
    tl.store(
        Out + output_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc,
    )
    tl.store(M + batch_head * N_CTX + offs_m, m_i)

    if ENABLE_JVP:
        tangent_probability_value = (
            tangent_acc / l_i[:, None] - (tangent_mass / l_i)[:, None] * acc
        )
        tangent_output = tangent_probability_value + tangent_value_acc / l_i[:, None]
        tangent_output_offset = (
            batch_idx.to(tl.int64) * stride_toz + head_idx.to(tl.int64) * stride_toh
        )
        tl.store(
            TangentOut
            + tangent_output_offset
            + offs_m[:, None] * stride_tom
            + offs_d[None, :] * stride_tod,
            tangent_output,
        )


@triton.jit
def _attn_fwd_sparse_tma(
    sm_scale,
    M,
    desc_q,
    desc_k,
    desc_v,
    desc_tangent_q,
    desc_tangent_k,
    desc_tangent_v,
    desc_out,
    desc_tangent_out,
    PartialKVNumBlocks,
    PartialKVIndices,
    PartialKVMaskIds,
    FullKVNumBlocks,
    FullKVIndices,
    PartialMasks,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MASK_BATCH: tl.constexpr,
    MASK_HEADS: tl.constexpr,
    MAX_PARTIAL: tl.constexpr,
    MAX_FULL: tl.constexpr,
    ENABLE_JVP: tl.constexpr,
):
    """Descriptor-backed sparse fused primal/JVP attention forward."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // H
    head_idx = batch_head % H
    mask_batch_idx = batch_idx if MASK_BATCH > 1 else 0
    mask_head_idx = head_idx if MASK_HEADS > 1 else 0
    metadata_row = (mask_batch_idx * MASK_HEADS + mask_head_idx) * N_BLOCKS + query_block
    sequence_offset = batch_head * N_CTX
    query_offset = sequence_offset + query_block * _TRITON_BLOCK_SIZE
    q = desc_q.load([query_offset, 0])
    acc = tl.zeros((_TRITON_BLOCK_SIZE, HEAD_DIM), tl.float32)
    l_i = tl.full((_TRITON_BLOCK_SIZE,), 1.0, tl.float32)
    m_i = tl.full((_TRITON_BLOCK_SIZE,), -float("inf"), tl.float32)

    if ENABLE_JVP:
        tangent_q = desc_tangent_q.load([query_offset, 0])
        tangent_acc = tl.zeros((_TRITON_BLOCK_SIZE, HEAD_DIM), tl.float32)
        tangent_mass = tl.zeros((_TRITON_BLOCK_SIZE,), tl.float32)
        tangent_value_acc = tl.zeros((_TRITON_BLOCK_SIZE, HEAD_DIM), tl.float32)
    else:
        tangent_q = q
        tangent_acc = tl.zeros((1, 1), tl.float32)
        tangent_mass = tl.zeros((1,), tl.float32)
        tangent_value_acc = tl.zeros((1, 1), tl.float32)

    qk_scale = sm_scale * 1.4426950408889634
    full_count = tl.load(FullKVNumBlocks + metadata_row)
    full_row_offset = metadata_row.to(tl.int64) * MAX_FULL
    for entry_idx in range(full_count):
        kv_block = tl.load(FullKVIndices + full_row_offset + entry_idx)
        (
            acc,
            tangent_acc,
            l_i,
            m_i,
            tangent_mass,
            tangent_value_acc,
        ) = _sparse_fwd_tile_tma(
            acc,
            tangent_acc,
            l_i,
            m_i,
            tangent_mass,
            tangent_value_acc,
            q,
            tangent_q,
            desc_k,
            desc_v,
            desc_tangent_k,
            desc_tangent_v,
            PartialMasks,
            sequence_offset,
            kv_block,
            0,
            sm_scale,
            qk_scale,
            HEAD_DIM,
            ENABLE_JVP,
            IS_PARTIAL=False,
        )

    partial_count = tl.load(PartialKVNumBlocks + metadata_row)
    partial_row_offset = metadata_row.to(tl.int64) * MAX_PARTIAL
    for entry_idx in range(partial_count):
        kv_block = tl.load(PartialKVIndices + partial_row_offset + entry_idx)
        mask_id = tl.load(PartialKVMaskIds + partial_row_offset + entry_idx)
        (
            acc,
            tangent_acc,
            l_i,
            m_i,
            tangent_mass,
            tangent_value_acc,
        ) = _sparse_fwd_tile_tma(
            acc,
            tangent_acc,
            l_i,
            m_i,
            tangent_mass,
            tangent_value_acc,
            q,
            tangent_q,
            desc_k,
            desc_v,
            desc_tangent_k,
            desc_tangent_v,
            PartialMasks,
            sequence_offset,
            kv_block,
            mask_id,
            sm_scale,
            qk_scale,
            HEAD_DIM,
            ENABLE_JVP,
            IS_PARTIAL=True,
        )

    m_i += tl.math.log2(l_i)
    acc /= l_i[:, None]
    desc_out.store([query_offset, 0], acc.to(desc_out.dtype))
    offs_m = query_block * _TRITON_BLOCK_SIZE + tl.arange(0, _TRITON_BLOCK_SIZE)
    tl.store(M + batch_head * N_CTX + offs_m, m_i)
    if ENABLE_JVP:
        tangent_probability_value = (
            tangent_acc / l_i[:, None] - (tangent_mass / l_i)[:, None] * acc
        )
        tangent_output = tangent_probability_value + tangent_value_acc / l_i[:, None]
        desc_tangent_out.store([query_offset, 0], tangent_output.to(desc_tangent_out.dtype))


@triton.jit
def _sparse_bwd_preprocess(
    Out,
    DOut,
    Delta,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Compute ``sum(output * doutput)`` for each query row."""
    block_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    offs_m = block_idx * _TRITON_BLOCK_SIZE + tl.arange(0, _TRITON_BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    base = batch_head * N_CTX * HEAD_DIM
    out = tl.load(Out + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
    dout = tl.load(DOut + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]).to(tl.float32)
    tl.store(
        Delta + batch_head * N_CTX + offs_m,
        tl.sum(out * dout, axis=1),
    )


@triton.jit
def _sparse_bwd_dkdv_tile(
    dk,
    dv,
    Q,
    DOut,
    M,
    Delta,
    PartialMasks,
    qkv_offset,
    delta_offset,
    k,
    v,
    query_block,
    mask_id,
    qk_scale,
    stride_tok,
    stride_d,
    HEAD_DIM: tl.constexpr,
    IS_PARTIAL: tl.constexpr,
):
    """Accumulate one query tile into dK/dV for a fixed KV tile."""
    offs_m = query_block * _TRITON_BLOCK_SIZE + tl.arange(0, _TRITON_BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + qkv_offset + offs_m[:, None] * stride_tok + offs_d[None, :] * stride_d)
    dout = tl.load(DOut + qkv_offset + offs_m[:, None] * stride_tok + offs_d[None, :] * stride_d)
    logsumexp = tl.load(M + delta_offset + offs_m)
    delta = tl.load(Delta + delta_offset + offs_m)

    score_t = tl.dot(k, tl.trans(q)) * qk_scale - logsumexp[None, :]
    if IS_PARTIAL:
        mask_offsets = (
            mask_id.to(tl.int64) * _TRITON_BLOCK_SIZE * _TRITON_BLOCK_SIZE
            + tl.arange(0, _TRITON_BLOCK_SIZE)[None, :] * _TRITON_BLOCK_SIZE
            + tl.arange(0, _TRITON_BLOCK_SIZE)[:, None]
        )
        element_mask_t = tl.load(PartialMasks + mask_offsets)
        score_t = tl.where(element_mask_t, score_t, -float("inf"))
    probability_t = tl.math.exp2(score_t)

    dv += tl.dot(probability_t.to(tl.float32), dout.to(tl.float32))
    d_probability_t = tl.dot(v, tl.trans(dout)).to(tl.float32)
    d_score_t = probability_t * (d_probability_t - delta[None, :])
    dk += tl.dot(d_score_t.to(tl.float32), q.to(tl.float32))
    return dk, dv


@triton.jit
def _sparse_bwd_dq_tile(
    dq,
    K,
    V,
    PartialMasks,
    qkv_offset,
    q,
    dout,
    logsumexp,
    delta,
    kv_block,
    mask_id,
    qk_scale,
    stride_tok,
    stride_d,
    HEAD_DIM: tl.constexpr,
    IS_PARTIAL: tl.constexpr,
):
    """Accumulate one KV tile into dQ for a fixed query tile."""
    offs_n = kv_block * _TRITON_BLOCK_SIZE + tl.arange(0, _TRITON_BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    k_t = tl.load(K + qkv_offset + offs_d[:, None] * stride_d + offs_n[None, :] * stride_tok)
    v_t = tl.load(V + qkv_offset + offs_d[:, None] * stride_d + offs_n[None, :] * stride_tok)
    score = tl.dot(q, k_t) * qk_scale - logsumexp[:, None]
    if IS_PARTIAL:
        mask_offsets = (
            mask_id.to(tl.int64) * _TRITON_BLOCK_SIZE * _TRITON_BLOCK_SIZE
            + tl.arange(0, _TRITON_BLOCK_SIZE)[:, None] * _TRITON_BLOCK_SIZE
            + tl.arange(0, _TRITON_BLOCK_SIZE)[None, :]
        )
        element_mask = tl.load(PartialMasks + mask_offsets)
        score = tl.where(element_mask, score, -float("inf"))
    probability = tl.math.exp2(score)

    d_probability = tl.dot(dout, v_t).to(tl.float32)
    d_score = probability * (d_probability - delta[:, None])
    dq += tl.dot(d_score.to(tl.float32), tl.trans(k_t).to(tl.float32))
    return dq


@triton.jit
def _attn_bwd_sparse(
    Q,
    K,
    V,
    DOut,
    DQ,
    DK,
    DV,
    M,
    Delta,
    PartialKVNumBlocks,
    PartialKVIndices,
    PartialKVMaskIds,
    FullKVNumBlocks,
    FullKVIndices,
    PartialQNumBlocks,
    PartialQIndices,
    PartialQMaskIds,
    FullQNumBlocks,
    FullQIndices,
    PartialMasks,
    sm_scale,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MASK_BATCH: tl.constexpr,
    MASK_HEADS: tl.constexpr,
    MAX_PARTIAL_KV: tl.constexpr,
    MAX_FULL_KV: tl.constexpr,
    MAX_PARTIAL_Q: tl.constexpr,
    MAX_FULL_Q: tl.constexpr,
):
    """Sparse dQ, dK, and dV with one owner program per output tile."""
    block_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // H
    head_idx = batch_head % H
    mask_batch_idx = batch_idx if MASK_BATCH > 1 else 0
    mask_head_idx = head_idx if MASK_HEADS > 1 else 0
    metadata_row = (mask_batch_idx * MASK_HEADS + mask_head_idx) * N_BLOCKS + block_idx
    qkv_offset = batch_idx.to(tl.int64) * stride_z + head_idx.to(tl.int64) * stride_h
    delta_offset = batch_head * N_CTX
    offs_d = tl.arange(0, HEAD_DIM)
    qk_scale = sm_scale * 1.4426950408889634

    # dK and dV: this program owns KV block ``block_idx`` and traverses the
    # transposed sparse schedule of query blocks.
    offs_n = block_idx * _TRITON_BLOCK_SIZE + tl.arange(0, _TRITON_BLOCK_SIZE)
    k = tl.load(K + qkv_offset + offs_n[:, None] * stride_tok + offs_d[None, :] * stride_d)
    v = tl.load(V + qkv_offset + offs_n[:, None] * stride_tok + offs_d[None, :] * stride_d)
    dk = tl.zeros((_TRITON_BLOCK_SIZE, HEAD_DIM), tl.float32)
    dv = tl.zeros((_TRITON_BLOCK_SIZE, HEAD_DIM), tl.float32)

    full_q_count = tl.load(FullQNumBlocks + metadata_row)
    full_q_row_offset = metadata_row.to(tl.int64) * MAX_FULL_Q
    for entry_idx in range(full_q_count):
        query_block = tl.load(FullQIndices + full_q_row_offset + entry_idx)
        dk, dv = _sparse_bwd_dkdv_tile(
            dk,
            dv,
            Q,
            DOut,
            M,
            Delta,
            PartialMasks,
            qkv_offset,
            delta_offset,
            k,
            v,
            query_block,
            0,
            qk_scale,
            stride_tok,
            stride_d,
            HEAD_DIM,
            IS_PARTIAL=False,
        )

    partial_q_count = tl.load(PartialQNumBlocks + metadata_row)
    partial_q_row_offset = metadata_row.to(tl.int64) * MAX_PARTIAL_Q
    for entry_idx in range(partial_q_count):
        query_block = tl.load(PartialQIndices + partial_q_row_offset + entry_idx)
        mask_id = tl.load(PartialQMaskIds + partial_q_row_offset + entry_idx)
        dk, dv = _sparse_bwd_dkdv_tile(
            dk,
            dv,
            Q,
            DOut,
            M,
            Delta,
            PartialMasks,
            qkv_offset,
            delta_offset,
            k,
            v,
            query_block,
            mask_id,
            qk_scale,
            stride_tok,
            stride_d,
            HEAD_DIM,
            IS_PARTIAL=True,
        )

    tl.store(
        DK + qkv_offset + offs_n[:, None] * stride_tok + offs_d[None, :] * stride_d,
        dk * sm_scale,
    )
    tl.store(
        DV + qkv_offset + offs_n[:, None] * stride_tok + offs_d[None, :] * stride_d,
        dv,
    )

    # dQ: the same program also owns query block ``block_idx`` and traverses
    # the row-oriented sparse schedule of KV blocks.
    offs_m = block_idx * _TRITON_BLOCK_SIZE + tl.arange(0, _TRITON_BLOCK_SIZE)
    q = tl.load(Q + qkv_offset + offs_m[:, None] * stride_tok + offs_d[None, :] * stride_d)
    dout = tl.load(DOut + qkv_offset + offs_m[:, None] * stride_tok + offs_d[None, :] * stride_d)
    logsumexp = tl.load(M + delta_offset + offs_m)
    delta = tl.load(Delta + delta_offset + offs_m)
    dq = tl.zeros((_TRITON_BLOCK_SIZE, HEAD_DIM), tl.float32)

    full_kv_count = tl.load(FullKVNumBlocks + metadata_row)
    full_kv_row_offset = metadata_row.to(tl.int64) * MAX_FULL_KV
    for entry_idx in range(full_kv_count):
        kv_block = tl.load(FullKVIndices + full_kv_row_offset + entry_idx)
        dq = _sparse_bwd_dq_tile(
            dq,
            K,
            V,
            PartialMasks,
            qkv_offset,
            q,
            dout,
            logsumexp,
            delta,
            kv_block,
            0,
            qk_scale,
            stride_tok,
            stride_d,
            HEAD_DIM,
            IS_PARTIAL=False,
        )

    partial_kv_count = tl.load(PartialKVNumBlocks + metadata_row)
    partial_kv_row_offset = metadata_row.to(tl.int64) * MAX_PARTIAL_KV
    for entry_idx in range(partial_kv_count):
        kv_block = tl.load(PartialKVIndices + partial_kv_row_offset + entry_idx)
        mask_id = tl.load(PartialKVMaskIds + partial_kv_row_offset + entry_idx)
        dq = _sparse_bwd_dq_tile(
            dq,
            K,
            V,
            PartialMasks,
            qkv_offset,
            q,
            dout,
            logsumexp,
            delta,
            kv_block,
            mask_id,
            qk_scale,
            stride_tok,
            stride_d,
            HEAD_DIM,
            IS_PARTIAL=True,
        )

    tl.store(
        DQ + qkv_offset + offs_m[:, None] * stride_tok + offs_d[None, :] * stride_d,
        dq * sm_scale,
    )


def _strides_4d(tensor: Tensor) -> tuple[int, int, int, int]:
    """Return statically typed strides for a rank-four tensor."""
    return tuple(tensor.stride())  # type: ignore[return-value]


def supports_sparse_tma(device: torch.device) -> bool:
    """Return whether host tensor descriptors are usable on ``device``."""
    return (
        TensorDescriptor is not None
        and torch.cuda.is_available()
        and torch.version.cuda is not None
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 9
    )


def _launch_sparse_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_t: Tensor | None,
    k_t: Tensor | None,
    v_t: Tensor | None,
    block_mask: BlockSparseMask,
    sm_scale: float,
    *,
    use_tma: bool,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Allocate outputs and launch the sparse forward kernel."""
    batch, heads, sequence, head_dim = q.shape
    enable_jvp = q_t is not None
    if enable_jvp and (k_t is None or v_t is None):
        raise ValueError("Sparse JVP requires q_t, k_t, and v_t together.")

    out = torch.empty_like(q)
    tangent_out = torch.empty_like(q_t) if q_t is not None else None
    memory = torch.empty((batch, heads, sequence), device=q.device, dtype=torch.float32)
    tangent_q_arg = q if q_t is None else q_t
    tangent_k_arg = k if k_t is None else k_t
    tangent_v_arg = v if v_t is None else v_t
    tangent_out_arg = out if tangent_out is None else tangent_out

    grid = (sequence // BLOCK_SIZE, batch * heads)
    tma_enabled = use_tma and supports_sparse_tma(q.device)
    launch_options: dict[str, int] = {
        "num_warps": 4,
        "num_stages": 1 if head_dim >= 128 else 2,
    }
    if (
        not tma_enabled
        and q.is_cuda
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(q.device)[0] >= 10
    ):
        if enable_jvp:
            # The six 32xHEAD_DIM primal/tangent accumulators at HEAD_DIM=256
            # otherwise spill just beyond Blackwell's shared-memory limit.
            launch_options["maxnreg"] = 232 if head_dim >= 128 else 168
        else:
            launch_options["maxnreg"] = 96

    common_meta = {
        "H": heads,
        "N_CTX": sequence,
        "N_BLOCKS": sequence // BLOCK_SIZE,
        "HEAD_DIM": head_dim,
        "MASK_BATCH": block_mask.mask_batch_size,
        "MASK_HEADS": block_mask.mask_heads,
        "MAX_PARTIAL": block_mask.partial_kv_indices.shape[-1],
        "MAX_FULL": block_mask.full_kv_indices.shape[-1],
        "ENABLE_JVP": enable_jvp,
    }
    if tma_enabled:
        assert TensorDescriptor is not None
        descriptor_shape = [batch * heads * sequence, head_dim]
        descriptor_strides = [head_dim, 1]
        descriptor_block = [BLOCK_SIZE, head_dim]

        def descriptor(tensor: Tensor):
            return TensorDescriptor(
                tensor,
                shape=descriptor_shape,
                strides=descriptor_strides,
                block_shape=descriptor_block,
            )

        _attn_fwd_sparse_tma[grid](
            sm_scale,
            memory,
            descriptor(q),
            descriptor(k),
            descriptor(v),
            descriptor(tangent_q_arg),
            descriptor(tangent_k_arg),
            descriptor(tangent_v_arg),
            descriptor(out),
            descriptor(tangent_out_arg),
            block_mask.partial_kv_num_blocks,
            block_mask.partial_kv_indices,
            block_mask.partial_kv_mask_ids,
            block_mask.full_kv_num_blocks,
            block_mask.full_kv_indices,
            block_mask.partial_masks,
            **common_meta,
            **launch_options,
        )
        return out, tangent_out, memory

    _attn_fwd_sparse[grid](
        q,
        k,
        v,
        tangent_q_arg,
        tangent_k_arg,
        tangent_v_arg,
        out,
        tangent_out_arg,
        memory,
        block_mask.partial_kv_num_blocks,
        block_mask.partial_kv_indices,
        block_mask.partial_kv_mask_ids,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
        block_mask.partial_masks,
        sm_scale,
        *_strides_4d(q),
        *_strides_4d(k),
        *_strides_4d(v),
        *_strides_4d(tangent_q_arg),
        *_strides_4d(tangent_k_arg),
        *_strides_4d(tangent_v_arg),
        *_strides_4d(out),
        *_strides_4d(tangent_out_arg),
        **common_meta,
        **launch_options,
    )
    return out, tangent_out, memory


def sparse_attention_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    memory: Tensor,
    dout: Tensor,
    block_mask: BlockSparseMask,
    sm_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Launch sparse primal reverse mode for dQ, dK, and dV."""
    batch, heads, sequence, head_dim = q.shape
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    delta = torch.empty_like(memory)

    grid = (sequence // BLOCK_SIZE, batch * heads)
    launch_options: dict[str, int] = {"num_warps": 4, "num_stages": 2}
    if (
        q.is_cuda
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(q.device)[0] >= 10
    ):
        launch_options["maxnreg"] = 168

    _sparse_bwd_preprocess[grid](
        out,
        dout,
        delta,
        N_CTX=sequence,
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    _attn_bwd_sparse[grid](
        q,
        k,
        v,
        dout,
        dq,
        dk,
        dv,
        memory,
        delta,
        block_mask.partial_kv_num_blocks,
        block_mask.partial_kv_indices,
        block_mask.partial_kv_mask_ids,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
        block_mask.partial_q_num_blocks,
        block_mask.partial_q_indices,
        block_mask.partial_q_mask_ids,
        block_mask.full_q_num_blocks,
        block_mask.full_q_indices,
        block_mask.partial_masks,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        H=heads,
        N_CTX=sequence,
        N_BLOCKS=sequence // BLOCK_SIZE,
        HEAD_DIM=head_dim,
        MASK_BATCH=block_mask.mask_batch_size,
        MASK_HEADS=block_mask.mask_heads,
        MAX_PARTIAL_KV=block_mask.partial_kv_indices.shape[-1],
        MAX_FULL_KV=block_mask.full_kv_indices.shape[-1],
        MAX_PARTIAL_Q=block_mask.partial_q_indices.shape[-1],
        MAX_FULL_Q=block_mask.full_q_indices.shape[-1],
        **launch_options,
    )
    return dq, dk, dv


def _rebuild_block_mask(
    q: Tensor,
    metadata: list[Tensor] | tuple[Tensor, ...],
) -> BlockSparseMask:
    """Rebuild the lightweight Python wrapper around compiled-op tensor inputs."""
    return BlockSparseMask._from_tensor_values(
        q.shape[2],
        q.shape[2],
        BLOCK_SIZE,
        metadata,
    )


@torch.library.custom_op("jvp_flash_attention::attn_fwd_sparse_triton", mutates_args=())
def _attn_fwd_sparse_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    metadata: list[Tensor],
    sm_scale: float,
    use_tma: bool,
    enable_jvp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Opaque sparse primal/JVP launch for ``torch.compile``."""
    block_mask = _rebuild_block_mask(
        q,
        metadata,
    )
    out, tangent_out, memory = _launch_sparse_forward(
        q,
        k,
        v,
        q_t if enable_jvp else None,
        k_t if enable_jvp else None,
        v_t if enable_jvp else None,
        block_mask,
        sm_scale,
        use_tma=use_tma,
    )
    if tangent_out is None:
        tangent_out = torch.empty(0, device=q.device)
    return out, tangent_out, memory


@_attn_fwd_sparse_triton.register_fake
def _attn_fwd_sparse_triton_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    metadata: list[Tensor],
    sm_scale: float,
    use_tma: bool,
    enable_jvp: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Propagate sparse custom-op output shapes for FakeTensor execution."""
    return (
        torch.empty_like(q),
        torch.empty_like(q_t) if enable_jvp else torch.empty(0, device=q.device),
        torch.empty(q.shape[:3], device=q.device, dtype=torch.float32),
    )


@torch.library.custom_op("jvp_flash_attention::attn_bwd_sparse_triton", mutates_args=())
def _attn_bwd_sparse_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    memory: Tensor,
    dout: Tensor,
    metadata: list[Tensor],
    sm_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Opaque sparse reverse launch used by compiled autograd."""
    block_mask = _rebuild_block_mask(
        q,
        metadata,
    )
    return sparse_attention_backward(
        q,
        k,
        v,
        out,
        memory,
        dout.contiguous(),
        block_mask,
        sm_scale,
    )


@_attn_bwd_sparse_triton.register_fake
def _attn_bwd_sparse_triton_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    memory: Tensor,
    dout: Tensor,
    metadata: list[Tensor],
    sm_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Propagate sparse backward output shapes for FakeTensor execution."""
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_sparse_autograd(ctx, inputs, output) -> None:
    """Save sparse primal state needed by the custom-op backward."""
    out, tangent_out, memory = output
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(tangent_out, memory)
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2], out, memory, *inputs[6])
    ctx.sm_scale = inputs[7]


def _backward_sparse(ctx, grad_out, grad_tangent_out, _grad_memory):
    """Differentiate the sparse custom op through its primal output."""
    if grad_tangent_out is not None:
        raise NotImplementedError(
            "Reverse-mode differentiation through the JVP tangent is unsupported; "
            "detach the tangent before including it in a loss."
        )
    if grad_out is None:
        return None, None, None, None, None, None, [None] * 11, None, None, None
    q, k, v, out, memory, *metadata = ctx.saved_tensors
    dq, dk, dv = _attn_bwd_sparse_triton(
        q,
        k,
        v,
        out,
        memory,
        grad_out,
        metadata,
        ctx.sm_scale,
    )
    return dq, dk, dv, None, None, None, [None] * 11, None, None, None


_attn_fwd_sparse_triton.register_autograd(
    _backward_sparse,
    setup_context=_setup_sparse_autograd,
)


def sparse_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_t: Tensor | None,
    k_t: Tensor | None,
    v_t: Tensor | None,
    block_mask: BlockSparseMask,
    sm_scale: float,
    *,
    compiling: bool,
    use_tma: bool,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Dispatch the eager or opaque compiled sparse forward."""
    if compiling:
        enable_jvp = q_t is not None
        if enable_jvp != (k_t is not None) or enable_jvp != (v_t is not None):
            raise ValueError("Q/K/V tangents must either all be present or all be absent.")
        out, compiled_tangent, memory = _attn_fwd_sparse_triton(
            q,
            k,
            v,
            q if q_t is None else q_t,
            k if k_t is None else k_t,
            v if v_t is None else v_t,
            block_mask._tensor_values(),
            sm_scale,
            use_tma,
            enable_jvp,
        )
        return out, compiled_tangent if enable_jvp else None, memory
    return _launch_sparse_forward(
        q,
        k,
        v,
        q_t,
        k_t,
        v_t,
        block_mask,
        sm_scale,
        use_tma=use_tma,
    )

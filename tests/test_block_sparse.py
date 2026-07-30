from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from jvp_flash_attention import (
    AttentionImplementation,
    BlockSparseMask,
    flash_attention,
)
from tests.jit_sandbox_masks import PMFMaskConfig, make_pmf_mask_from_config


def _mask_patterns() -> dict[str, torch.Tensor]:
    length = 70
    all_full = torch.ones(64, 64, dtype=torch.bool)
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    block_diagonal = torch.zeros(64, 64, dtype=torch.bool)
    block_diagonal[:32, :32] = True
    block_diagonal[32:, 32:] = True
    sliding = torch.zeros(96, 96, dtype=torch.bool)
    for query in range(96):
        sliding[query, max(0, query - 39) : query + 1] = True
    random_full_tiles = torch.zeros(96, 96, dtype=torch.bool)
    random_full_tiles[:, :32] = True
    random_full_tiles[32:64, 64:96] = True
    one_partial = torch.eye(64, dtype=torch.bool)
    partial_edges = torch.eye(length, dtype=torch.bool)
    partial_edges[:32, 32:64] = True
    return {
        "all_full": all_full,
        "causal": causal,
        "block_diagonal": block_diagonal,
        "sliding": sliding,
        "random_full_tiles": random_full_tiles,
        "one_partial": one_partial,
        "partial_edges": partial_edges,
    }


@pytest.mark.parametrize("pattern", tuple(_mask_patterns()))
def test_mask_pattern_round_trips(pattern: str) -> None:
    mask = _mask_patterns()[pattern]
    block_mask = BlockSparseMask.from_bool(mask)

    assert torch.equal(block_mask.to_dense()[0, 0], mask)


def test_from_bool_round_trip_and_padding() -> None:
    length = 70
    mask = torch.zeros(length, length, dtype=torch.bool)
    mask[:32, :32] = True
    mask[32:64, 16:64] = True
    mask[64:, 64:] = torch.eye(6, dtype=torch.bool)

    block_mask = BlockSparseMask.from_bool(mask)

    assert block_mask.original_length == 70
    assert block_mask.padded_length == 96
    assert block_mask.block_size == 32
    assert torch.equal(block_mask.to_dense()[0, 0], mask)

    padded = block_mask.to_dense(padded=True)[0, 0]
    assert not padded[:length, length:].any()
    assert torch.equal(
        padded[length:, length:],
        torch.eye(block_mask.padded_length - length, dtype=torch.bool),
    )
    assert block_mask.num_full_tiles == 2
    assert block_mask.num_partial_tiles == 2


def test_batch_and_head_specific_round_trip() -> None:
    mask = torch.zeros(2, 3, 64, 64, dtype=torch.bool)
    for batch in range(2):
        for head in range(3):
            width = 8 * (head + 1)
            mask[batch, head, :, :width] = True
            mask[batch, head].diagonal().fill_(True)

    block_mask = BlockSparseMask.from_bool(mask)

    assert block_mask.mask_batch_size == 2
    assert block_mask.mask_heads == 3
    assert torch.equal(block_mask.to_dense(), mask)


def test_pytree_and_to_preserve_mask() -> None:
    dense = torch.eye(35, dtype=torch.bool)
    block_mask = BlockSparseMask.from_bool(dense)
    tensors, spec = torch.utils._pytree.tree_flatten(block_mask)
    rebuilt = torch.utils._pytree.tree_unflatten(tensors, spec)

    assert len(tensors) == 11
    assert torch.equal(rebuilt.to_dense(), block_mask.to_dense())
    assert torch.equal(block_mask.to("cpu").to_dense(), block_mask.to_dense())


def test_pad_to_capacity_preserves_mask_and_validates() -> None:
    dense = _mask_patterns()["partial_edges"]
    block_mask = BlockSparseMask.from_bool(dense)

    padded = block_mask.pad_to_capacity(
        schedule_width=block_mask.num_blocks,
        partial_mask_count=block_mask.num_blocks**2,
    )

    padded.validate()
    assert padded.partial_kv_indices.shape[-1] == block_mask.num_blocks
    assert padded.full_q_indices.shape[-1] == block_mask.num_blocks
    assert padded.partial_masks.shape[0] == block_mask.num_blocks**2
    assert torch.equal(padded.to_dense(), dense[None, None])


def test_pad_to_capacity_stabilizes_tensor_shapes_across_masks() -> None:
    dense_masks = (
        _mask_patterns()["all_full"],
        _mask_patterns()["one_partial"],
    )
    padded_masks = tuple(
        BlockSparseMask.from_bool(dense).pad_to_capacity(
            schedule_width=2,
            partial_mask_count=2,
        )
        for dense in dense_masks
    )

    tensor_shapes = tuple(
        tuple(tensor.shape for tensor in torch.utils._pytree.tree_leaves(block_mask))
        for block_mask in padded_masks
    )
    assert tensor_shapes[0] == tensor_shapes[1]
    for dense, block_mask in zip(dense_masks, padded_masks, strict=True):
        assert torch.equal(block_mask.to_dense(), dense[None, None])


def test_pad_to_capacity_rejects_small_capacities() -> None:
    block_mask = BlockSparseMask.from_bool(_mask_patterns()["partial_edges"])

    with pytest.raises(ValueError, match="schedule_width"):
        block_mask.pad_to_capacity(
            schedule_width=0,
            partial_mask_count=block_mask.num_blocks**2,
        )
    with pytest.raises(ValueError, match="partial_mask_count"):
        block_mask.pad_to_capacity(
            schedule_width=block_mask.num_blocks,
            partial_mask_count=0,
        )


def test_public_implementation_requires_matching_mask_kind() -> None:
    q = torch.empty(1, 1, 32, 16)
    block_mask = BlockSparseMask.from_bool(torch.eye(32, dtype=torch.bool))

    with pytest.raises(ValueError, match="does not accept block_mask"):
        flash_attention(
            q,
            q,
            q,
            implementation=AttentionImplementation.DENSE_POINTER,
            block_mask=block_mask,
        )
    with pytest.raises(TypeError, match="requires a BlockSparseMask"):
        flash_attention(
            q,
            q,
            q,
            implementation=AttentionImplementation.BLOCK_SPARSE_POINTER,
        )
    with pytest.raises(ValueError, match="does not accept attn_mask"):
        flash_attention(
            q,
            q,
            q,
            implementation=AttentionImplementation.BLOCK_SPARSE_POINTER,
            attn_mask=torch.eye(32, dtype=torch.bool),
            block_mask=block_mask,
        )
    with pytest.raises(ValueError, match="through block_mask"):
        flash_attention(
            q,
            q,
            q,
            implementation=AttentionImplementation.BLOCK_SPARSE_POINTER,
            block_mask=block_mask,
            causal=True,
        )


def test_construction_is_deterministic_and_schedules_are_transposes() -> None:
    dense = _mask_patterns()["partial_edges"]
    first = BlockSparseMask.from_bool(dense)
    second = BlockSparseMask.from_bool(dense.clone())
    first_tensors, _ = torch.utils._pytree.tree_flatten(first)
    second_tensors, _ = torch.utils._pytree.tree_flatten(second)
    for first_tensor, second_tensor in zip(first_tensors, second_tensors, strict=True):
        assert torch.equal(first_tensor, second_tensor)

    forward_full: set[tuple[int, int]] = set()
    transpose_full: set[tuple[int, int]] = set()
    forward_partial: set[tuple[int, int, int]] = set()
    transpose_partial: set[tuple[int, int, int]] = set()
    for row in range(first.num_blocks):
        full_count = int(first.full_kv_num_blocks[0, 0, row])
        full_indices = first.full_kv_indices[0, 0, row, :full_count].tolist()
        assert len(full_indices) == len(set(full_indices))
        forward_full.update((row, int(column)) for column in full_indices)

        partial_count = int(first.partial_kv_num_blocks[0, 0, row])
        partial_indices = first.partial_kv_indices[0, 0, row, :partial_count].tolist()
        partial_ids = first.partial_kv_mask_ids[0, 0, row, :partial_count].tolist()
        assert len(partial_indices) == len(set(partial_indices))
        forward_partial.update(
            (row, int(column), int(mask_id))
            for column, mask_id in zip(partial_indices, partial_ids, strict=True)
        )

        full_count = int(first.full_q_num_blocks[0, 0, row])
        full_indices = first.full_q_indices[0, 0, row, :full_count].tolist()
        assert len(full_indices) == len(set(full_indices))
        transpose_full.update((int(query), row) for query in full_indices)

        partial_count = int(first.partial_q_num_blocks[0, 0, row])
        partial_indices = first.partial_q_indices[0, 0, row, :partial_count].tolist()
        partial_ids = first.partial_q_mask_ids[0, 0, row, :partial_count].tolist()
        assert len(partial_indices) == len(set(partial_indices))
        transpose_partial.update(
            (int(query), row, int(mask_id))
            for query, mask_id in zip(partial_indices, partial_ids, strict=True)
        )

    assert forward_full == transpose_full
    assert forward_partial == transpose_partial
    assert forward_full.isdisjoint({edge[:2] for edge in forward_partial})
    for query, key, mask_id in forward_partial:
        tile = first.partial_masks[mask_id]
        assert tile.any() and not tile.all()
        assert 0 <= query < first.num_blocks
        assert 0 <= key < first.num_blocks


@pytest.mark.parametrize(
    ("mask", "error"),
    [
        (torch.ones(4, 4), TypeError),
        (torch.ones(3, 4, dtype=torch.bool), ValueError),
        (torch.zeros(4, 4, dtype=torch.bool), ValueError),
        (torch.ones(1, 1, 1, dtype=torch.bool), ValueError),
        (torch.ones(0, 1, 4, 4, dtype=torch.bool), ValueError),
    ],
)
def test_invalid_masks_are_rejected(mask: torch.Tensor, error: type[Exception]) -> None:
    with pytest.raises(error):
        BlockSparseMask.from_bool(mask)


def test_invalid_block_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="block_size=32"):
        BlockSparseMask.from_bool(torch.eye(32, dtype=torch.bool), block_size=64)


def test_validate_rejects_corrupt_schedule_metadata() -> None:
    block_mask = BlockSparseMask.from_bool(torch.ones(64, 64, dtype=torch.bool))
    corrupt_indices = block_mask.full_kv_indices.clone()
    corrupt_indices[0, 0, 0, 0] = block_mask.num_blocks
    corrupt = replace(block_mask, full_kv_indices=corrupt_indices)

    with pytest.raises(ValueError, match="out-of-bounds"):
        corrupt.validate()


def test_validate_rejects_an_empty_query_row() -> None:
    block_mask = BlockSparseMask.from_bool(torch.eye(32, dtype=torch.bool))
    corrupt_payloads = block_mask.partial_masks.clone()
    corrupt_payloads[0, 0] = False
    corrupt = replace(block_mask, partial_masks=corrupt_payloads)

    with pytest.raises(ValueError, match="Every query row"):
        corrupt.validate()


def test_validate_rejects_invalid_padding_visibility() -> None:
    block_mask = BlockSparseMask.from_bool(torch.eye(35, dtype=torch.bool))
    corrupt_payloads = block_mask.partial_masks.clone()
    final_tile_id = int(block_mask.partial_kv_mask_ids[0, 0, 1, 0])

    # Real query 32 must not see padded key 63.
    corrupt_payloads[final_tile_id, 0, 31] = True
    corrupt = replace(block_mask, partial_masks=corrupt_payloads)
    with pytest.raises(ValueError, match="padded keys"):
        corrupt.validate()

    # Padded query 35 must attend only to its own padded key.
    corrupt_payloads = block_mask.partial_masks.clone()
    corrupt_payloads[final_tile_id, 3, 0] = True
    corrupt = replace(block_mask, partial_masks=corrupt_payloads)
    with pytest.raises(ValueError, match="padded rows"):
        corrupt.validate()


def test_realistic_pmf_fixture_matches_recorded_geometry() -> None:
    fixture = make_pmf_mask_from_config(PMFMaskConfig())
    block_mask = BlockSparseMask.from_bool(fixture.mask)

    assert fixture.sequence_length == 3896
    assert fixture.clean_tokens == 1976
    assert fixture.mask.float().mean().item() == pytest.approx(0.10129485)
    assert block_mask.padded_length == 3904
    assert block_mask.num_full_tiles == 1354
    assert block_mask.num_partial_tiles == 312
    assert block_mask.tile_density == pytest.approx(0.1119323)
    assert block_mask.storage_bytes < 2 * 1024 * 1024
    expanded_storage = 12 * block_mask.padded_length * block_mask.padded_length
    assert expanded_storage / block_mask.storage_bytes > 50
    assert torch.equal(block_mask.to_dense()[0, 0], fixture.mask)


@pytest.mark.parametrize(
    (
        "mode",
        "registers",
        "sequence",
        "allowed_elements",
        "padded",
        "full_tiles",
        "partial_tiles",
    ),
    [
        ("chunk", 0, 3840, 1_572_864, 3840, 1536, 0),
        ("row", 0, 3896, 1_488_640, 3904, 1307, 258),
        ("ring", 0, 3876, 1_505_056, 3904, 1307, 469),
        ("hilbert_4", 0, 3900, 1_482_624, 3904, 1307, 258),
        ("hilbert_16", 0, 3888, 1_500_672, 3904, 1307, 258),
        ("hilbert_64", 0, 3840, 1_572_864, 3840, 1536, 0),
        ("hilbert_random", 0, 3896, 1_537_536, 3904, 1354, 312),
        ("hilbert_random", 1, 4168, 1_747_693, 4192, 1429, 647),
        ("hilbert_random", 4, 4984, 2_460_496, 4992, 2088, 720),
    ],
)
def test_realistic_pmf_matrix_exact_statistics(
    mode: str,
    registers: int,
    sequence: int,
    allowed_elements: int,
    padded: int,
    full_tiles: int,
    partial_tiles: int,
) -> None:
    fixture = make_pmf_mask_from_config(
        PMFMaskConfig(mode=mode, register_tokens_per_block=registers)
    )
    block_mask = BlockSparseMask.from_bool(fixture.mask)

    assert fixture.sequence_length == sequence
    assert int(fixture.mask.sum()) == allowed_elements
    assert block_mask.padded_length == padded
    assert block_mask.num_full_tiles == full_tiles
    assert block_mask.num_partial_tiles == partial_tiles

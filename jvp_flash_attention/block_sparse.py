"""Tensor-only block-sparse attention mask metadata.

The Triton kernels consume two views of the same sparse tile graph:

* ``*_kv_*`` lists key/value tiles for each query tile (forward and dQ);
* ``*_q_*`` lists query tiles for each key/value tile (dK and dV).

Completely full tiles do not carry an element mask. Partially full tiles share a
compact 32x32 Boolean payload between the two traversal directions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Any

import torch
from torch import Tensor
from torch.utils import _pytree

_TENSOR_FIELDS = (
    "partial_kv_num_blocks",
    "partial_kv_indices",
    "partial_kv_mask_ids",
    "full_kv_num_blocks",
    "full_kv_indices",
    "partial_q_num_blocks",
    "partial_q_indices",
    "partial_q_mask_ids",
    "full_q_num_blocks",
    "full_q_indices",
    "partial_masks",
)


def _make_schedule(
    rows: list[list[list[list[tuple[int, int] | int]]]],
    *,
    with_mask_ids: bool,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Pack nested ``[batch][head][row]`` lists into padded int32 tensors."""
    batch = len(rows)
    heads = len(rows[0])
    n_blocks = len(rows[0][0])
    max_blocks = max(
        1,
        max(
            len(rows[b][h][row])
            for b in range(batch)
            for h in range(heads)
            for row in range(n_blocks)
        ),
    )

    counts = torch.zeros((batch, heads, n_blocks), dtype=torch.int32)
    indices = torch.zeros((batch, heads, n_blocks, max_blocks), dtype=torch.int32)
    mask_ids = (
        torch.zeros((batch, heads, n_blocks, max_blocks), dtype=torch.int32)
        if with_mask_ids
        else None
    )

    for batch_idx in range(batch):
        for head_idx in range(heads):
            for row_idx in range(n_blocks):
                entries = rows[batch_idx][head_idx][row_idx]
                counts[batch_idx, head_idx, row_idx] = len(entries)
                for entry_idx, entry in enumerate(entries):
                    if with_mask_ids:
                        index, mask_id = entry
                        indices[batch_idx, head_idx, row_idx, entry_idx] = index
                        assert mask_ids is not None
                        mask_ids[batch_idx, head_idx, row_idx, entry_idx] = mask_id
                    else:
                        indices[batch_idx, head_idx, row_idx, entry_idx] = entry

    return counts, indices, mask_ids


def _pad_last_dimension(value: Tensor, capacity: int) -> Tensor:
    if value.shape[-1] == capacity:
        return value
    padding = value.new_zeros(*value.shape[:-1], capacity - value.shape[-1])
    return torch.cat([value, padding], dim=-1)


def _pad_first_dimension(value: Tensor, capacity: int) -> Tensor:
    if value.shape[0] == capacity:
        return value
    padding = value.new_zeros(capacity - value.shape[0], *value.shape[1:])
    return torch.cat([value, padding], dim=0)


@dataclass(frozen=True, eq=False)
class BlockSparseMask:
    """Precomputed 32x32 block-sparse Boolean attention mask.

    Mask batch and head dimensions may be one, meaning that the schedule is
    broadcast across the corresponding Q/K/V dimension. Construct masks with
    ``from_bool`` and treat the tensor fields as immutable metadata.
    """

    original_length: int
    padded_length: int
    block_size: int

    partial_kv_num_blocks: Tensor
    partial_kv_indices: Tensor
    partial_kv_mask_ids: Tensor
    full_kv_num_blocks: Tensor
    full_kv_indices: Tensor

    partial_q_num_blocks: Tensor
    partial_q_indices: Tensor
    partial_q_mask_ids: Tensor
    full_q_num_blocks: Tensor
    full_q_indices: Tensor

    partial_masks: Tensor

    def _tensor_values(self) -> list[Tensor]:
        """Return tensor fields in their stable pytree order."""
        return [getattr(self, name) for name in _TENSOR_FIELDS]

    @classmethod
    def _from_tensor_values(
        cls,
        original_length: int,
        padded_length: int,
        block_size: int,
        tensors: list[Tensor] | tuple[Tensor, ...],
    ) -> BlockSparseMask:
        """Rebuild a mask from its scalar metadata and pytree tensor leaves."""
        if len(tensors) != len(_TENSOR_FIELDS):
            raise ValueError(f"Expected {len(_TENSOR_FIELDS)} mask tensors, got {len(tensors)}.")
        return cls(original_length, padded_length, block_size, *tensors)

    @classmethod
    def from_bool(cls, mask: Tensor, *, block_size: int = 32) -> BlockSparseMask:
        """Compress a square Boolean mask into sparse tile schedules.

        ``mask`` may have shape ``[sequence, sequence]`` or
        ``[mask_batch, mask_heads, sequence, sequence]``. A mask dimension of
        one is broadcast when the mask is used.
        """
        if block_size != 32:
            raise ValueError(f"Only block_size=32 is currently supported, got {block_size}.")
        if mask.dtype != torch.bool:
            raise TypeError(f"BlockSparseMask requires a Boolean tensor, got {mask.dtype}.")
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim != 4:
            raise ValueError(
                "BlockSparseMask expects [N, N] or [mask_batch, mask_heads, N, N], "
                f"got shape {tuple(mask.shape)}."
            )
        if mask.shape[-2] != mask.shape[-1]:
            raise ValueError(f"BlockSparseMask requires a square mask, got {tuple(mask.shape)}.")
        if mask.shape[0] == 0 or mask.shape[1] == 0:
            raise ValueError("BlockSparseMask batch and head dimensions must be non-empty.")

        original_length = mask.shape[-1]
        if original_length == 0:
            raise ValueError("BlockSparseMask requires a non-empty sequence.")
        device = mask.device
        mask = mask.detach().cpu()
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("Every real query row must attend to at least one key.")

        batch, heads = mask.shape[:2]
        padded_length = ((original_length + block_size - 1) // block_size) * block_size
        n_blocks = padded_length // block_size

        # This dense tensor is temporary construction workspace only. The
        # compressed object returned below does not retain it.
        padded = torch.zeros(
            (batch, heads, padded_length, padded_length),
            dtype=torch.bool,
        )
        padded[..., :original_length, :original_length] = mask
        if padded_length != original_length:
            pad_indices = torch.arange(original_length, padded_length)
            padded[..., pad_indices, pad_indices] = True

        tiles = (
            padded.reshape(batch, heads, n_blocks, block_size, n_blocks, block_size)
            .permute(0, 1, 2, 4, 3, 5)
            .contiguous()
        )
        any_tiles = tiles.any(dim=(-1, -2))
        full_tiles = tiles.all(dim=(-1, -2))

        def empty_rows() -> list[list[list[list[Any]]]]:
            return [[[[] for _ in range(n_blocks)] for _ in range(heads)] for _ in range(batch)]

        partial_kv = empty_rows()
        full_kv = empty_rows()
        partial_q = empty_rows()
        full_q = empty_rows()
        partial_payloads: list[Tensor] = []

        for batch_idx in range(batch):
            for head_idx in range(heads):
                for query_block in range(n_blocks):
                    for kv_block in range(n_blocks):
                        if not bool(any_tiles[batch_idx, head_idx, query_block, kv_block]):
                            continue
                        if bool(full_tiles[batch_idx, head_idx, query_block, kv_block]):
                            full_kv[batch_idx][head_idx][query_block].append(kv_block)
                            full_q[batch_idx][head_idx][kv_block].append(query_block)
                        else:
                            mask_id = len(partial_payloads)
                            partial_payloads.append(
                                tiles[batch_idx, head_idx, query_block, kv_block].clone()
                            )
                            partial_kv[batch_idx][head_idx][query_block].append(
                                (kv_block, mask_id)
                            )
                            partial_q[batch_idx][head_idx][kv_block].append((query_block, mask_id))

        (
            partial_kv_num_blocks,
            partial_kv_indices,
            partial_kv_mask_ids,
        ) = _make_schedule(partial_kv, with_mask_ids=True)
        full_kv_num_blocks, full_kv_indices, _ = _make_schedule(full_kv, with_mask_ids=False)
        (
            partial_q_num_blocks,
            partial_q_indices,
            partial_q_mask_ids,
        ) = _make_schedule(partial_q, with_mask_ids=True)
        full_q_num_blocks, full_q_indices, _ = _make_schedule(full_q, with_mask_ids=False)

        if partial_payloads:
            partial_masks = torch.stack(partial_payloads)
        else:
            # A valid pointer is convenient for Triton even when every count is
            # zero. No schedule references this sentinel payload.
            partial_masks = torch.zeros((1, block_size, block_size), dtype=torch.bool)

        assert partial_kv_mask_ids is not None
        assert partial_q_mask_ids is not None
        result = cls(
            original_length=original_length,
            padded_length=padded_length,
            block_size=block_size,
            partial_kv_num_blocks=partial_kv_num_blocks,
            partial_kv_indices=partial_kv_indices,
            partial_kv_mask_ids=partial_kv_mask_ids,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            partial_q_num_blocks=partial_q_num_blocks,
            partial_q_indices=partial_q_indices,
            partial_q_mask_ids=partial_q_mask_ids,
            full_q_num_blocks=full_q_num_blocks,
            full_q_indices=full_q_indices,
            partial_masks=partial_masks,
        )
        result.validate()
        return result if device.type == "cpu" else result.to(device)

    @property
    def device(self) -> torch.device:
        """Return the device shared by all schedule tensors."""
        return self.partial_kv_num_blocks.device

    @property
    def mask_batch_size(self) -> int:
        """Return the mask batch dimension, which may broadcast from one."""
        return self.partial_kv_num_blocks.shape[0]

    @property
    def mask_heads(self) -> int:
        """Return the mask head dimension, which may broadcast from one."""
        return self.partial_kv_num_blocks.shape[1]

    @property
    def num_blocks(self) -> int:
        """Return the number of padded query and key tiles."""
        return self.padded_length // self.block_size

    @property
    def num_partial_tiles(self) -> int:
        """Return the number of scheduled element-masked tiles."""
        return int(self.partial_kv_num_blocks.sum().item())

    @property
    def num_full_tiles(self) -> int:
        """Return the number of scheduled fully visible tiles."""
        return int(self.full_kv_num_blocks.sum().item())

    @property
    def storage_bytes(self) -> int:
        """Return storage occupied by tensor metadata and partial payloads."""
        return sum(
            getattr(self, name).numel() * getattr(self, name).element_size()
            for name in _TENSOR_FIELDS
        )

    def pad_to_capacity(
        self,
        *,
        schedule_width: int,
        partial_mask_count: int,
    ) -> BlockSparseMask:
        """Pad inactive metadata slots to fixed capacities without changing the mask."""
        schedule_width = int(schedule_width)
        partial_mask_count = int(partial_mask_count)
        schedule_tensors = (
            self.partial_kv_indices,
            self.partial_kv_mask_ids,
            self.full_kv_indices,
            self.partial_q_indices,
            self.partial_q_mask_ids,
            self.full_q_indices,
        )
        minimum_schedule_width = max(tensor.shape[-1] for tensor in schedule_tensors)
        if schedule_width < minimum_schedule_width:
            raise ValueError(
                f"schedule_width must be at least {minimum_schedule_width}, "
                f"got {schedule_width}."
            )
        if partial_mask_count < self.partial_masks.shape[0]:
            raise ValueError(
                f"partial_mask_count must be at least {self.partial_masks.shape[0]}, "
                f"got {partial_mask_count}."
            )
        result = replace(
            self,
            partial_kv_indices=_pad_last_dimension(
                self.partial_kv_indices, schedule_width
            ),
            partial_kv_mask_ids=_pad_last_dimension(
                self.partial_kv_mask_ids, schedule_width
            ),
            full_kv_indices=_pad_last_dimension(self.full_kv_indices, schedule_width),
            partial_q_indices=_pad_last_dimension(
                self.partial_q_indices, schedule_width
            ),
            partial_q_mask_ids=_pad_last_dimension(
                self.partial_q_mask_ids, schedule_width
            ),
            full_q_indices=_pad_last_dimension(self.full_q_indices, schedule_width),
            partial_masks=_pad_first_dimension(
                self.partial_masks, partial_mask_count
            ),
        )
        result.validate()
        return result

    @property
    def tile_density(self) -> float:
        """Return the fraction of full or partial tiles in the padded grid."""
        denominator = self.mask_batch_size * self.mask_heads * self.num_blocks**2
        return (self.num_partial_tiles + self.num_full_tiles) / denominator

    def validate(self) -> None:
        """Validate tensor shapes, devices, dtypes, and schedule bounds."""
        if self.block_size != 32:
            raise ValueError(f"Only block_size=32 is supported, got {self.block_size}.")
        if self.original_length <= 0 or self.original_length > self.padded_length:
            raise ValueError(
                "Expected 0 < original_length <= padded_length, got "
                f"{self.original_length} and {self.padded_length}."
            )
        if self.padded_length % self.block_size:
            raise ValueError("padded_length must be divisible by block_size.")
        if self.partial_kv_num_blocks.ndim != 3:
            raise ValueError("Sparse count tensors must have rank 3.")
        if self.mask_batch_size == 0 or self.mask_heads == 0:
            raise ValueError("Sparse batch and head dimensions must be non-empty.")

        devices = {getattr(self, name).device for name in _TENSOR_FIELDS}
        if len(devices) != 1:
            raise ValueError(f"All BlockSparseMask tensors must share a device, got {devices}.")
        if any(not getattr(self, name).is_contiguous() for name in _TENSOR_FIELDS):
            raise ValueError("All BlockSparseMask tensors must be contiguous.")

        base_shape = (
            self.mask_batch_size,
            self.mask_heads,
            self.num_blocks,
        )
        for name in (
            "partial_kv_num_blocks",
            "full_kv_num_blocks",
            "partial_q_num_blocks",
            "full_q_num_blocks",
        ):
            tensor = getattr(self, name)
            if tensor.dtype != torch.int32 or tensor.shape != base_shape:
                raise ValueError(
                    f"{name} must be int32 with shape {base_shape}, "
                    f"got {tensor.dtype} {tuple(tensor.shape)}."
                )

        for name in (
            "partial_kv_indices",
            "partial_kv_mask_ids",
            "full_kv_indices",
            "partial_q_indices",
            "partial_q_mask_ids",
            "full_q_indices",
        ):
            tensor = getattr(self, name)
            if tensor.dtype != torch.int32 or tensor.shape[:3] != base_shape or tensor.ndim != 4:
                raise ValueError(
                    f"{name} must be a rank-4 int32 schedule beginning with {base_shape}, "
                    f"got {tensor.dtype} {tuple(tensor.shape)}."
                )

        if (
            self.partial_masks.dtype != torch.bool
            or self.partial_masks.ndim != 3
            or self.partial_masks.shape[0] == 0
            or self.partial_masks.shape[1:] != (self.block_size, self.block_size)
        ):
            raise ValueError(
                "partial_masks must be Boolean [tiles, block_size, block_size], got "
                f"{self.partial_masks.dtype} {tuple(self.partial_masks.shape)}."
            )

        host = {name: getattr(self, name).detach().cpu() for name in _TENSOR_FIELDS}
        if not bool((host["partial_kv_num_blocks"] + host["full_kv_num_blocks"] > 0).all()):
            raise ValueError("Every query tile must schedule at least one KV tile.")

        def schedule_edges(
            name: str,
            counts_name: str,
            indices_name: str,
            mask_ids_name: str | None = None,
            *,
            transposed: bool = False,
        ) -> set[tuple[int, int, int, int, int | None]]:
            counts = host[counts_name]
            indices = host[indices_name]
            mask_ids = None if mask_ids_name is None else host[mask_ids_name]
            edges: set[tuple[int, int, int, int, int | None]] = set()
            for batch, head, row in product(
                range(self.mask_batch_size),
                range(self.mask_heads),
                range(self.num_blocks),
            ):
                count = int(counts[batch, head, row])
                if not 0 <= count <= indices.shape[-1] or (
                    mask_ids is not None and count > mask_ids.shape[-1]
                ):
                    raise ValueError("Sparse schedule count exceeds its allocated row width.")
                active_indices = [
                    int(index) for index in indices[batch, head, row, :count].tolist()
                ]
                if len(active_indices) != len(set(active_indices)):
                    raise ValueError(f"{name} contains a duplicate tile in row {row}.")
                if any(not 0 <= index < self.num_blocks for index in active_indices):
                    raise ValueError(f"{name} contains an out-of-bounds tile index.")

                active_mask_ids = (
                    [None] * count
                    if mask_ids is None
                    else [int(mask_id) for mask_id in mask_ids[batch, head, row, :count].tolist()]
                )
                if any(
                    mask_id is not None and not 0 <= mask_id < self.partial_masks.shape[0]
                    for mask_id in active_mask_ids
                ):
                    raise ValueError(f"{name} contains an out-of-bounds partial mask ID.")
                for column, mask_id in zip(active_indices, active_mask_ids, strict=True):
                    query, key = (column, row) if transposed else (row, column)
                    edges.add((batch, head, query, key, mask_id))
            return edges

        forward_full = schedule_edges("full_kv", "full_kv_num_blocks", "full_kv_indices")
        transpose_full = schedule_edges(
            "full_q", "full_q_num_blocks", "full_q_indices", transposed=True
        )
        forward_partial = schedule_edges(
            "partial_kv",
            "partial_kv_num_blocks",
            "partial_kv_indices",
            "partial_kv_mask_ids",
        )
        transpose_partial = schedule_edges(
            "partial_q",
            "partial_q_num_blocks",
            "partial_q_indices",
            "partial_q_mask_ids",
            transposed=True,
        )

        if forward_full != transpose_full:
            raise ValueError("Full forward and transposed schedules disagree.")
        if forward_partial != transpose_partial:
            raise ValueError("Partial forward and transposed schedules disagree.")
        if {edge[:4] for edge in forward_full} & {edge[:4] for edge in forward_partial}:
            raise ValueError("A tile cannot be both full and partial.")
        forward_mask_ids = [mask_id for *_, mask_id in forward_partial if mask_id is not None]
        if len(forward_mask_ids) != len(set(forward_mask_ids)):
            raise ValueError("Each partial tile must own one partial mask ID.")
        if forward_mask_ids:
            expected_ids = set(range(len(forward_mask_ids)))
            if set(forward_mask_ids) != expected_ids:
                raise ValueError("Partial mask IDs must form a compact range.")
            for mask_id in expected_ids:
                payload = host["partial_masks"][mask_id]
                if not bool(payload.any()) or bool(payload.all()):
                    raise ValueError(
                        "Each scheduled partial payload must be nonempty and not full."
                    )
        if self.partial_masks.shape[0] < max(1, len(forward_mask_ids)):
            raise ValueError("Partial mask storage does not cover every scheduled mask ID.")
        inactive_payloads = host["partial_masks"][len(forward_mask_ids) :]
        if bool(inactive_payloads.any()):
            raise ValueError("Inactive partial mask payloads must be zero.")

        row_has_key = torch.zeros(
            (self.mask_batch_size, self.mask_heads, self.padded_length),
            dtype=torch.bool,
        )
        offsets = torch.arange(self.block_size)

        def padding_allowed(query_block: int, key_block: int) -> Tensor:
            query_positions = query_block * self.block_size + offsets[:, None]
            key_positions = key_block * self.block_size + offsets[None, :]
            return (
                (query_positions < self.original_length) & (key_positions < self.original_length)
            ) | ((query_positions >= self.original_length) & (query_positions == key_positions))

        for batch, head, query_block, key_block, _ in forward_full:
            query_start = query_block * self.block_size
            if not bool(padding_allowed(query_block, key_block).all()):
                raise ValueError(
                    "Full tiles must not expose padded keys or non-diagonal padded rows."
                )
            row_has_key[
                batch,
                head,
                query_start : query_start + self.block_size,
            ] = True

        for batch, head, query_block, key_block, mask_id in forward_partial:
            assert mask_id is not None
            payload = host["partial_masks"][mask_id]
            query_start = query_block * self.block_size
            if bool((payload & ~padding_allowed(query_block, key_block)).any()):
                raise ValueError(
                    "Partial tiles must not expose padded keys or non-diagonal padded rows."
                )
            row_has_key[
                batch,
                head,
                query_start : query_start + self.block_size,
            ] |= payload.any(dim=-1)

        if not bool(row_has_key.all()):
            raise ValueError("Every query row must attend to at least one key.")

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> BlockSparseMask:
        """Move all mask metadata tensors to ``device``."""
        return replace(
            self,
            **{
                name: getattr(self, name).to(device=device, non_blocking=non_blocking)
                for name in _TENSOR_FIELDS
            },
        )

    def to_dense(self, *, padded: bool = False) -> Tensor:
        """Reconstruct the Boolean mask for testing and debugging."""
        if self.device.type != "cpu":
            return self.to("cpu").to_dense(padded=padded).to(self.device)
        dense = torch.zeros(
            (
                self.mask_batch_size,
                self.mask_heads,
                self.padded_length,
                self.padded_length,
            ),
            dtype=torch.bool,
            device=self.device,
        )
        block = self.block_size
        for batch_idx in range(self.mask_batch_size):
            for head_idx in range(self.mask_heads):
                for query_block in range(self.num_blocks):
                    q_slice = slice(query_block * block, (query_block + 1) * block)
                    full_count = int(
                        self.full_kv_num_blocks[batch_idx, head_idx, query_block].item()
                    )
                    for entry_idx in range(full_count):
                        kv_block = int(
                            self.full_kv_indices[
                                batch_idx, head_idx, query_block, entry_idx
                            ].item()
                        )
                        k_slice = slice(kv_block * block, (kv_block + 1) * block)
                        dense[batch_idx, head_idx, q_slice, k_slice] = True

                    partial_count = int(
                        self.partial_kv_num_blocks[batch_idx, head_idx, query_block].item()
                    )
                    for entry_idx in range(partial_count):
                        kv_block = int(
                            self.partial_kv_indices[
                                batch_idx, head_idx, query_block, entry_idx
                            ].item()
                        )
                        mask_id = int(
                            self.partial_kv_mask_ids[
                                batch_idx, head_idx, query_block, entry_idx
                            ].item()
                        )
                        k_slice = slice(kv_block * block, (kv_block + 1) * block)
                        dense[batch_idx, head_idx, q_slice, k_slice] = self.partial_masks[mask_id]
        if padded:
            return dense
        return dense[..., : self.original_length, : self.original_length]


def _flatten_mask(mask: BlockSparseMask) -> tuple[list[Tensor], tuple[int, int, int]]:
    """Flatten a mask into tensor leaves plus immutable scalar context."""
    return (
        mask._tensor_values(),
        (mask.original_length, mask.padded_length, mask.block_size),
    )


def _unflatten_mask(
    tensors: list[Tensor],
    context: tuple[int, int, int],
) -> BlockSparseMask:
    """Rebuild a mask from pytree tensor leaves and scalar context."""
    original_length, padded_length, block_size = context
    return BlockSparseMask._from_tensor_values(
        original_length,
        padded_length,
        block_size,
        tensors,
    )


_pytree.register_pytree_node(
    BlockSparseMask,
    _flatten_mask,
    _unflatten_mask,
    serialized_type_name="jvp_flash_attention.BlockSparseMask",
)

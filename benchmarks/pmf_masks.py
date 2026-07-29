"""Self-contained pixel mean-flow block-causal mask fixtures.

This reproduces the packed clean/noisy stream geometry used by jit_sandbox without importing that
repository. The defaults correspond to a 120-frame target plus one observed tubelet at 256x256 with
4x32x32 tubelets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

BlockMode = Literal[
    "chunk",
    "row",
    "ring",
    "hilbert_4",
    "hilbert_16",
    "hilbert_64",
    "hilbert_random",
]


@dataclass(frozen=True)
class PMFMaskFixture:
    """Dense Boolean fixture plus its packed-stream metadata."""

    mask: Tensor
    packed_token_ids: Tensor
    packed_block_ids: Tensor
    clean_tokens: int
    block_sizes: tuple[int, ...]
    block_start_chunks: Tensor
    mode: str
    history_chunks: int
    register_tokens_per_block: int
    seed: int

    @property
    def sequence_length(self) -> int:
        return self.mask.shape[-1]


@dataclass(frozen=True)
class PMFMaskConfig:
    """Video-level inputs for the local jit_sandbox mask reproduction."""

    image_height: int = 256
    image_width: int = 256
    target_frames: int = 120
    temporal_patch: int = 4
    spatial_patch_height: int = 32
    spatial_patch_width: int = 32
    condition_chunks: int = 1
    mode: BlockMode = "hilbert_random"
    register_tokens_per_block: int = 0
    history_chunks: int = 6
    seed: int = 0

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        if self.target_frames % self.temporal_patch:
            raise ValueError("target_frames must be divisible by temporal_patch.")
        if self.image_height % self.spatial_patch_height:
            raise ValueError("image_height must be divisible by spatial_patch_height.")
        if self.image_width % self.spatial_patch_width:
            raise ValueError("image_width must be divisible by spatial_patch_width.")
        return (
            self.condition_chunks + self.target_frames // self.temporal_patch,
            self.image_height // self.spatial_patch_height,
            self.image_width // self.spatial_patch_width,
        )

    @property
    def target_start_token(self) -> int:
        _, grid_h, grid_w = self.grid_shape
        return self.condition_chunks * grid_h * grid_w


def _block_sizes(
    mode: BlockMode,
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    seed: int,
) -> tuple[int, ...]:
    spatial_tokens = grid_h * grid_w
    if mode == "chunk":
        return (spatial_tokens,) * grid_t
    if mode == "row":
        return (grid_w,) * (grid_t * grid_h)
    if mode == "ring":
        if (grid_h, grid_w) != (8, 8):
            raise ValueError("The local ring fixture currently supports the realistic 8x8 grid.")
        # Centre-to-edge ring order; the final 28-token outer ring matches the
        # production packed-stream geometry.
        return (4, 12, 20, 28) * grid_t
    if mode in {"hilbert_4", "hilbert_16", "hilbert_64"}:
        size = int(mode.removeprefix("hilbert_"))
        return (size,) * (grid_t * spatial_tokens // size)
    if mode != "hilbert_random":
        raise ValueError(f"Unsupported pMF block mode {mode!r}.")

    choices = tuple(
        size
        for size in (4, 8, 16, 32, 64)
        if size <= spatial_tokens and spatial_tokens % size == 0
    )
    if not choices:
        raise ValueError("Hilbert fixtures require at least four spatial tokens.")
    generator = torch.Generator().manual_seed(seed)
    choice_ids = torch.randint(len(choices), (grid_t,), generator=generator).tolist()
    result: list[int] = []
    for choice_id in choice_ids:
        size = choices[choice_id]
        result.extend((size,) * (spatial_tokens // size))
    return tuple(result)


def _pack_blocks(
    video_token_ids: Tensor,
    token_block_ids: Tensor,
    *,
    num_video_tokens: int,
    register_tokens_per_block: int,
    target_start_token: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Insert a shared register set before each selected complete block."""
    device = video_token_ids.device
    if video_token_ids.numel() == 0:
        empty_ids = torch.empty(0, dtype=torch.long, device=device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        return empty_ids, empty_ids, empty_bool

    selected_block_ids = token_block_ids.index_select(0, video_token_ids)
    unique_blocks = torch.unique_consecutive(selected_block_ids)
    packed_tokens: list[Tensor] = []
    packed_blocks: list[Tensor] = []
    packed_fixed: list[Tensor] = []
    for block_id_tensor in unique_blocks:
        block_id = int(block_id_tensor.item())
        in_block = selected_block_ids == block_id
        block_video_ids = video_token_ids[in_block]
        block_is_fixed = bool((block_video_ids < target_start_token).all())
        if register_tokens_per_block:
            register_types = torch.arange(
                register_tokens_per_block, dtype=torch.long, device=device
            )
            register_ids = num_video_tokens + block_id * register_tokens_per_block + register_types
            packed_tokens.append(register_ids)
            packed_blocks.append(torch.full_like(register_ids, block_id, dtype=torch.long))
            packed_fixed.append(
                torch.full(
                    (register_tokens_per_block,),
                    block_is_fixed,
                    dtype=torch.bool,
                    device=device,
                )
            )
        packed_tokens.append(block_video_ids)
        packed_blocks.append(torch.full_like(block_video_ids, block_id))
        packed_fixed.append(block_video_ids < target_start_token)
    return (
        torch.cat(packed_tokens),
        torch.cat(packed_blocks),
        torch.cat(packed_fixed),
    )


def make_pmf_mask(
    *,
    mode: BlockMode = "hilbert_random",
    seed: int = 0,
    register_tokens_per_block: int = 0,
    history_chunks: int = 6,
    grid_shape: tuple[int, int, int] = (31, 8, 8),
    target_start_token: int = 64,
    device: torch.device | str = "cpu",
) -> PMFMaskFixture:
    """Build a realistic pMF training attention mask.

    The clean stream contains the fixed prefix and all target blocks except the final semantic
    block. The noisy stream contains the complete target suffix.
    """
    grid_t, grid_h, grid_w = grid_shape
    num_video_tokens = grid_t * grid_h * grid_w
    if not 0 < target_start_token < num_video_tokens:
        raise ValueError("target_start_token must split a non-empty video sequence.")
    if register_tokens_per_block < 0:
        raise ValueError("register_tokens_per_block must be non-negative.")
    if history_chunks < 0:
        raise ValueError("history_chunks must be non-negative.")

    device = torch.device(device)
    block_sizes = _block_sizes(
        mode,
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
        seed=seed,
    )
    block_counts = torch.tensor(block_sizes, dtype=torch.long, device=device)
    block_offsets = torch.cumsum(block_counts, dim=0) - block_counts
    token_block_ids = torch.repeat_interleave(
        torch.arange(len(block_sizes), dtype=torch.long, device=device),
        block_counts,
    )
    block_start_chunks = torch.div(
        block_offsets,
        grid_h * grid_w,
        rounding_mode="floor",
    )

    fixed_ids = torch.arange(target_start_token, dtype=torch.long, device=device)
    target_ids = torch.arange(
        target_start_token, num_video_tokens, dtype=torch.long, device=device
    )
    target_blocks = token_block_ids.index_select(0, target_ids)
    target_context_ids = target_ids[target_blocks < len(block_sizes) - 1]
    clean_video_ids = torch.cat((fixed_ids, target_context_ids))

    clean_ids, clean_blocks, clean_is_fixed = _pack_blocks(
        clean_video_ids,
        token_block_ids,
        num_video_tokens=num_video_tokens,
        register_tokens_per_block=register_tokens_per_block,
        target_start_token=target_start_token,
    )
    noisy_ids, noisy_blocks, _ = _pack_blocks(
        target_ids,
        token_block_ids,
        num_video_tokens=num_video_tokens,
        register_tokens_per_block=register_tokens_per_block,
        target_start_token=target_start_token,
    )
    packed_token_ids = torch.cat((clean_ids, noisy_ids))
    packed_block_ids = torch.cat((clean_blocks, noisy_blocks))
    clean_tokens = clean_ids.numel()

    clean_available = torch.zeros(len(block_sizes), dtype=torch.bool, device=device)
    clean_available[clean_blocks.unique()] = True
    packed_tokens = packed_token_ids.numel()
    indices = torch.arange(packed_tokens, dtype=torch.long, device=device)
    query_is_clean = indices[:, None] < clean_tokens
    key_is_clean = indices[None, :] < clean_tokens
    key_is_fixed = torch.cat(
        (
            clean_is_fixed,
            torch.zeros(noisy_ids.numel(), dtype=torch.bool, device=device),
        )
    )[None, :]
    q_block = packed_block_ids[:, None]
    k_block = packed_block_ids[None, :]
    q_token = packed_token_ids[:, None]
    k_token = packed_token_ids[None, :]

    query_start = block_start_chunks[q_block]
    key_start = block_start_chunks[k_block]
    visible = (k_block <= q_block) & (key_start >= query_start - history_chunks)
    query_is_noisy = ~query_is_clean
    key_is_noisy = ~key_is_clean
    same_block = q_block == k_block
    same_token = q_token == k_token
    clean_to_clean = query_is_clean & key_is_clean & visible
    noisy_to_clean = (
        query_is_noisy & key_is_clean & visible & (key_is_fixed | ((~same_block) & (~same_token)))
    )
    noisy_to_noisy = (
        query_is_noisy & key_is_noisy & visible & (same_block | (~clean_available[k_block]))
    )
    mask = clean_to_clean | noisy_to_clean | noisy_to_noisy
    if not bool(mask.any(dim=-1).all()):
        raise RuntimeError("Generated pMF fixture contains an empty query row.")

    return PMFMaskFixture(
        mask=mask,
        packed_token_ids=packed_token_ids,
        packed_block_ids=packed_block_ids,
        clean_tokens=clean_tokens,
        block_sizes=block_sizes,
        block_start_chunks=block_start_chunks,
        mode=mode,
        history_chunks=history_chunks,
        register_tokens_per_block=register_tokens_per_block,
        seed=seed,
    )


def make_pmf_mask_from_config(
    config: PMFMaskConfig,
    *,
    device: torch.device | str = "cpu",
) -> PMFMaskFixture:
    """Build a fixture from video and tubelet dimensions."""
    return make_pmf_mask(
        mode=config.mode,
        seed=config.seed,
        register_tokens_per_block=config.register_tokens_per_block,
        history_chunks=config.history_chunks,
        grid_shape=config.grid_shape,
        target_start_token=config.target_start_token,
        device=device,
    )

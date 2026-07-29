from __future__ import annotations

import torch

from tests.jit_sandbox_masks import make_pmf_mask


def _small_chunk_fixture(*, history_chunks: int = 6, registers: int = 0):
    return make_pmf_mask(
        mode="chunk",
        grid_shape=(4, 2, 2),
        target_start_token=4,
        history_chunks=history_chunks,
        register_tokens_per_block=registers,
    )


def test_clean_and_noisy_stream_packing() -> None:
    fixture = _small_chunk_fixture()

    # The clean stream contains the observed block and target context, but
    # deliberately omits the final target block. The noisy stream contains the
    # entire target suffix.
    assert fixture.clean_tokens == 12
    assert torch.equal(fixture.packed_token_ids[:12], torch.arange(12))
    assert torch.equal(fixture.packed_token_ids[12:], torch.arange(4, 16))
    assert not (fixture.packed_token_ids[: fixture.clean_tokens] >= 12).any()


def test_block_causal_visibility_and_same_token_exclusion() -> None:
    fixture = _small_chunk_fixture()
    mask = fixture.mask
    clean = fixture.clean_tokens

    clean_block_two_query = 8
    assert mask[clean_block_two_query, :12].all()
    assert not mask[clean_block_two_query, clean:].any()

    noisy_block_one_query = clean
    assert mask[noisy_block_one_query, :4].all()  # observed fixed prefix
    assert not mask[noisy_block_one_query, 4:8].any()  # same clean block/token
    assert mask[noisy_block_one_query, clean : clean + 4].all()

    noisy_block_two_query = clean + 4
    assert not mask[noisy_block_two_query, clean : clean + 4].any()
    assert mask[noisy_block_two_query, clean + 4 : clean + 8].all()
    assert not mask[noisy_block_two_query, clean + 8 :].any()

    # The same video token appears once in each stream, but a noisy query must
    # not attend its clean counterpart.
    assert fixture.packed_token_ids[4] == fixture.packed_token_ids[clean]
    assert not mask[clean, 4]


def test_temporal_history_window_is_enforced() -> None:
    no_history = _small_chunk_fixture(history_chunks=0)
    q_index = no_history.clean_tokens + 8  # noisy query in final block
    q_block = int(no_history.packed_block_ids[q_index])
    visible_key_blocks = no_history.packed_block_ids[no_history.mask[q_index]]

    assert torch.equal(visible_key_blocks.unique(), torch.tensor([q_block]))


def test_registers_are_inserted_before_each_stream_block() -> None:
    fixture = _small_chunk_fixture(registers=2)
    clean_ids = fixture.packed_token_ids[: fixture.clean_tokens]
    noisy_ids = fixture.packed_token_ids[fixture.clean_tokens :]

    expected_clean = torch.tensor(
        [
            16,
            17,
            0,
            1,
            2,
            3,
            18,
            19,
            4,
            5,
            6,
            7,
            20,
            21,
            8,
            9,
            10,
            11,
        ]
    )
    expected_noisy = torch.tensor(
        [
            18,
            19,
            4,
            5,
            6,
            7,
            20,
            21,
            8,
            9,
            10,
            11,
            22,
            23,
            12,
            13,
            14,
            15,
        ]
    )
    assert torch.equal(clean_ids.cpu(), expected_clean)
    assert torch.equal(noisy_ids.cpu(), expected_noisy)

    noisy_block_one = fixture.clean_tokens
    # Registers belonging to the observed block are fixed context. Registers
    # from the query's own clean block remain excluded.
    assert fixture.mask[noisy_block_one, :2].all()
    assert not fixture.mask[noisy_block_one, 6:8].any()

import torch
import pytest

from ar_image_generation.approaches.var.multiscale import (
    build_multiscale_sequence,
    build_multiscale_token_grids,
    build_next_scale_attention_mask,
    build_scale_ids,
)
from ar_image_generation.approaches.var.schedule import VARSchedule


def test_var_schedule_properties() -> None:
    schedule = VARSchedule(scales=(1, 2, 4, 8))

    assert schedule.final_scale == 8
    assert schedule.num_scales == 4
    assert schedule.num_tokens == 1 + 4 + 16 + 64


def test_var_schedule_rejects_non_increasing_scales() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        VARSchedule(scales=(1, 4, 4, 8))


def test_build_multiscale_token_grids_shapes() -> None:
    schedule = VARSchedule(scales=(1, 2, 4, 8))
    tokens = torch.arange(2 * 8 * 8).reshape(2, 8, 8).long()

    grids = build_multiscale_token_grids(tokens, schedule)

    assert len(grids) == 4
    assert grids[0].shape == (2, 1, 1)
    assert grids[1].shape == (2, 2, 2)
    assert grids[2].shape == (2, 4, 4)
    assert grids[3].shape == (2, 8, 8)


def test_build_multiscale_sequence_shape() -> None:
    schedule = VARSchedule(scales=(1, 2, 4, 8))
    tokens = torch.randint(low=0, high=512, size=(2, 8, 8)).long()

    sequence = build_multiscale_sequence(tokens, schedule)

    assert sequence.shape == (2, 85)


def test_build_scale_ids_with_bos() -> None:
    schedule = VARSchedule(scales=(1, 2, 4))
    scale_ids = build_scale_ids(schedule, include_bos=True)

    assert scale_ids.shape == (1 + 1 + 4 + 16,)
    assert scale_ids[0].item() == -1
    assert scale_ids[1].item() == 0
    assert scale_ids[2].item() == 1
    assert scale_ids[-1].item() == 2


def test_next_scale_attention_mask_rules() -> None:
    schedule = VARSchedule(scales=(1, 2, 4))
    mask = build_next_scale_attention_mask(schedule, include_bos=True)

    # Total sequence length:
    # BOS + 1x1 + 2x2 + 4x4
    assert mask.shape == (22, 22)

    # BOS attends only to itself.
    assert mask[0, 0]
    assert not mask[0, 1:].any()

    # First 1x1 scale token attends to BOS only.
    assert mask[1, 0]
    assert not mask[1, 1]

    # First 2x2 scale token attends to BOS and previous 1x1 scale.
    assert mask[2, 0]
    assert mask[2, 1]

    # But it does not attend to same-scale 2x2 tokens.
    assert not mask[2, 2]
    assert not mask[2, 3]

    # First 4x4 scale token attends to previous scales.
    first_4x4_position = 1 + 1 + 4
    assert mask[first_4x4_position, 0]
    assert mask[first_4x4_position, 1]
    assert mask[first_4x4_position, 2]

    # But it does not attend to itself.
    assert not mask[first_4x4_position, first_4x4_position]

import pytest

from ar_image_generation.approaches.registry import build_approach


def test_unknown_approach_raises() -> None:
    with pytest.raises(ValueError, match="Unknown approach"):
        build_approach("does_not_exist")

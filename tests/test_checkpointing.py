import torch
from torch import nn

from ar_image_generation.engine.checkpointing import load_model_checkpoint, save_model_checkpoint


def test_save_and_load_model_checkpoint(tmp_path) -> None:
    model = nn.Linear(4, 2)
    checkpoint_path = tmp_path / "model.pt"

    save_model_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        epoch=3,
        metrics={"loss": 1.23},
        config={"name": "test"},
    )

    loaded_model = nn.Linear(4, 2)
    checkpoint = load_model_checkpoint(
        path=checkpoint_path,
        model=loaded_model,
        map_location="cpu",
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["metrics"]["loss"] == 1.23

    for original_parameter, loaded_parameter in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(original_parameter, loaded_parameter)

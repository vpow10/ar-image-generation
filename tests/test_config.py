from ar_image_generation.config import load_experiment_config


def test_load_experiment_config() -> None:
    cfg = load_experiment_config("configs/experiment/var_pathmnist64_debug.yaml")

    assert cfg.dataset.name == "pathmnist"
    assert cfg.dataset.size == 64
    assert cfg.dataset.batch_size == 32
    assert cfg.tokenizer.vocab_size == 512
    assert cfg.approach.name == "var"

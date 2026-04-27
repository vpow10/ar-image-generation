from medmnist import INFO

from ar_image_generation.data.medmnist import _resolve_medmnist_dataset_class


def test_pathmnist_exists_in_medmnist() -> None:
    assert "pathmnist" in INFO


def test_resolve_pathmnist_class() -> None:
    dataset_cls = _resolve_medmnist_dataset_class("pathmnist")

    assert dataset_cls.__name__ == "PathMNIST"

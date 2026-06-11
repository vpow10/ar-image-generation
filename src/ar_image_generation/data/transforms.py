from torchvision import transforms


class _NormalizeToMinusOneOne:
    """[0, 1] → [-1, 1]. Top-level class so DataLoader workers can pickle it."""

    def __call__(self, x):
        return x * 2.0 - 1.0


def build_image_transform(*, normalize: bool) -> transforms.Compose:
    transform_steps = [transforms.ToTensor()]

    if normalize:
        transform_steps.append(_NormalizeToMinusOneOne())

    return transforms.Compose(transform_steps)

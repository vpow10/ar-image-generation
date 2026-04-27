from torchvision import transforms


def build_image_transform(*, normalize: bool) -> transforms.Compose:
    transform_steps = [
        transforms.ToTensor(),
    ]

    if normalize:
        # [0, 1] -> [-1, 1]
        transform_steps.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))

    return transforms.Compose(transform_steps)

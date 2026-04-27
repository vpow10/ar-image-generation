from collections.abc import Callable

from ar_image_generation.approaches.base import AutoregressiveApproach

ApproachFactory = Callable[..., AutoregressiveApproach]

_APPROACH_REGISTRY: dict[str, ApproachFactory] = {}
_BUILTINS_IMPORTED = False


def register_approach(name: str) -> Callable[[ApproachFactory], ApproachFactory]:
    def decorator(factory: ApproachFactory) -> ApproachFactory:
        if name in _APPROACH_REGISTRY:
            raise ValueError(f"Approach already registered: {name}")
        _APPROACH_REGISTRY[name] = factory
        return factory

    return decorator


def import_builtin_approaches() -> None:
    global _BUILTINS_IMPORTED

    if _BUILTINS_IMPORTED:
        return

    # Importing modules triggers @register_approach decorators.
    import ar_image_generation.approaches.var.model  # noqa: F401

    _BUILTINS_IMPORTED = True


def build_approach(name: str, **kwargs: object) -> AutoregressiveApproach:
    import_builtin_approaches()

    if name not in _APPROACH_REGISTRY:
        available = ", ".join(sorted(_APPROACH_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown approach '{name}'. Available: {available}")

    return _APPROACH_REGISTRY[name](**kwargs)

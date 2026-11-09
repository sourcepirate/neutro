from ..base import Layer
from .dense import Dense


def select_dense_layers(model, target_modules):
    layers = model._get_all_layers()
    if target_modules is None:
        return [layer for layer in layers if isinstance(layer, Dense)]

    if isinstance(target_modules, list) and target_modules and all(
        isinstance(item, Layer) for item in target_modules
    ):
        return [layer for layer in target_modules if isinstance(layer, Dense)]

    patterns = [target_modules] if isinstance(target_modules, str) else target_modules
    if not isinstance(patterns, list):
        raise ValueError(f"Unsupported target_modules type: {type(target_modules)}")

    return [
        layer
        for layer in layers
        if isinstance(layer, Dense)
        and any(
            pattern in (layer.name or "") or pattern in type(layer).__name__
            for pattern in patterns
        )
    ]


def replace_layer(parent, old_layer, new_layer):
    for name, value in list(vars(parent).items()):
        if value is old_layer:
            setattr(parent, name, new_layer)
            return True

        if isinstance(value, list):
            for index, item in enumerate(value):
                if item is old_layer:
                    value[index] = new_layer
                    return True
                if isinstance(item, Layer) and replace_layer(item, old_layer, new_layer):
                    return True
            continue

        if isinstance(value, tuple):
            items = list(value)
            replaced = False
            for index, item in enumerate(items):
                if item is old_layer:
                    items[index] = new_layer
                    replaced = True
                elif isinstance(item, Layer) and replace_layer(item, old_layer, new_layer):
                    replaced = True
            if replaced:
                setattr(parent, name, tuple(items))
                return True
            continue

        if isinstance(value, dict):
            for key, item in list(value.items()):
                if item is old_layer:
                    value[key] = new_layer
                    return True
                if isinstance(item, Layer) and replace_layer(item, old_layer, new_layer):
                    return True
            continue

        if isinstance(value, Layer) and replace_layer(value, old_layer, new_layer):
            return True

    return False


def replace_in_model(model, old_layer, new_layer):
    return replace_layer(model, old_layer, new_layer)

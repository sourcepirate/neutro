"""Keras-style ``model.summary()`` table."""


def print_summary(model):
    """Print a layer/param table for ``model``.

    Functional models also show which layers each layer is connected to.
    """
    is_functional = model.inputs is not None

    print("-" * 85)
    if is_functional:
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<10} {'Connected to':<25}")
    else:
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<10}")
    print("=" * 85)

    total_params = 0
    trainable_params = 0

    curr_shape = None
    if model.layers and model.layers[0].input_shape:
        curr_shape = model.layers[0].input_shape

    for layer in model.layers:
        name = layer.name or layer.__class__.__name__
        layer_type = layer.__class__.__name__

        # Use the layer's own shapes when it has been built; otherwise fall
        # back to the shape inferred from the previous layer.
        if layer.built:
            try:
                output_shape = layer.compute_output_shape(layer.input_shape)
            except Exception:
                output_shape = "multiple"
        elif curr_shape is not None:
            try:
                output_shape = layer.compute_output_shape(curr_shape)
                curr_shape = output_shape
            except Exception:
                output_shape = "multiple"
        else:
            output_shape = "unbuilt"

        params = layer.count_params()
        total_params += params
        if getattr(layer, 'trainable', True):
            trainable_params += params

        if is_functional:
            connected_to = _connected_layers(model, layer)
            connected_str = ", ".join(connected_to) if connected_to else ""
            print(f"{f'{name} ({layer_type})':<25} {output_shape!s:<20} {params:<10,} {connected_str:<25}")
        else:
            print(f"{f'{name} ({layer_type})':<25} {output_shape!s:<20} {params:<10,}")

    print("=" * 85)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("-" * 85)


def _connected_layers(model, layer):
    """Names of layers feeding ``layer`` in a Functional model."""
    connected = []
    if not hasattr(layer, '_inbound_nodes'):
        return connected

    for node in layer._inbound_nodes:
        if node not in model._nodes_ordered:
            continue
        if isinstance(node.input_tensors, list):
            for tensor in node.input_tensors:
                if tensor.node:
                    connected.append(
                        tensor.node.layer.name or tensor.node.layer.__class__.__name__)
        elif node.input_tensors and node.input_tensors.node:
            connected.append(
                node.input_tensors.node.layer.name
                or node.input_tensors.node.layer.__class__.__name__)
    return connected

from ..base_model import Sequential
from ...layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Softmax


def _vgg_block(filters, num_convs, data_format):
    layers = [
        Conv2D(filters, kernel_size=3, padding='same', activation='relu', data_format=data_format)
        for _ in range(num_convs)
    ]
    layers.append(MaxPooling2D(pool_size=2, strides=2, data_format=data_format))
    return layers


def _vgg(input_shape, num_classes, data_format, block_convs):
    layers = []
    filters = 64
    for num_convs in block_convs:
        layers.extend(_vgg_block(filters, num_convs, data_format))
        filters *= 2

    model = Sequential([
        *layers,
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes),
        Softmax(),
    ])
    model.layers[0].input_shape = (None,) + tuple(input_shape)
    return model


def VGG16(input_shape=(224, 224, 3), num_classes=1000, data_format='channels_last'):
    """VGG16: five blocks of 3x3 convolutions followed by a classifier."""
    return _vgg(input_shape, num_classes, data_format, [2, 2, 3, 3, 3])


def VGG19(input_shape=(224, 224, 3), num_classes=1000, data_format='channels_last'):
    """VGG19: the deeper VGG16 variant with three extra convolutions."""
    return _vgg(input_shape, num_classes, data_format, [2, 2, 4, 4, 4])

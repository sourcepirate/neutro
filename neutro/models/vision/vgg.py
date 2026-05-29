from ..base_model import Sequential
from ...layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Softmax

def _vgg_block(filters, num_convs):
    layers = []
    for _ in range(num_convs):
        layers.append(Conv2D(filters, kernel_size=3, padding='same', activation='relu'))
    layers.append(MaxPooling2D(pool_size=2, strides=2))
    return layers

def VGG16(input_shape=(224, 224, 3), num_classes=1000):
    """
    VGG16: The "All 3x3" champion.
    The Oxford team decided that if one 3x3 kernel is good, 
    sixteen of them must be better. They weren't wrong.
    """
    layers = []
    # Block 1
    layers.extend(_vgg_block(64, 2))
    # Block 2
    layers.extend(_vgg_block(128, 2))
    # Block 3
    layers.extend(_vgg_block(256, 3))
    # Block 4
    layers.extend(_vgg_block(512, 3))
    # Block 5
    layers.extend(_vgg_block(512, 3))
    
    # Head
    model = Sequential([
        *layers,
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes),
        Softmax()
    ])
    
    # Set the input shape for the first layer
    model.layers[0].input_shape = input_shape
    return model

def VGG19(input_shape=(224, 224, 3), num_classes=1000):
    """VGG19: Because 16 layers just didn't feel deep enough."""
    layers = []
    layers.extend(_vgg_block(64, 2))
    layers.extend(_vgg_block(128, 2))
    layers.extend(_vgg_block(256, 4))
    layers.extend(_vgg_block(512, 4))
    layers.extend(_vgg_block(512, 4))
    
    model = Sequential([
        *layers,
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes),
        Softmax()
    ])
    model.layers[0].input_shape = input_shape
    return model

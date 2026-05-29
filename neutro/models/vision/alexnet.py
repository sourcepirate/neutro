from ..base_model import Sequential
from ...layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Softmax

def AlexNet(input_shape=(227, 227, 3), num_classes=1000):
    """
    AlexNet: The 2012 Grandfather of modern CNNs.
    Back then, we thought 11x11 kernels were a good idea. 
    It was a simpler time.
    """
    model = Sequential([
        # Layer 1: Large kernels for the "Big Picture"
        Conv2D(96, kernel_size=11, strides=4, activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=3, strides=2),
        
        # Layer 2
        Conv2D(256, kernel_size=5, padding='same', activation='relu'),
        MaxPooling2D(pool_size=3, strides=2),
        
        # Layer 3, 4, 5: The triple threat
        Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=3, strides=2),
        
        Flatten(),
        
        # The Fully Connected "Heads"
        # Note: Dropout was revolutionary here.
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes),
        Softmax()
    ])
    
    return model

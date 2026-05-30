import numpy as np
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neutro.models import Sequential
from neutro.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Softmax
from neutro.utils.data_utils import load_mnist
from neutro.preprocessing.image import ImageDataGenerator
from neutro.optimizers import Adam

def train_mnist():
    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Preprocess
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
    
    # One-hot encode labels
    y_train_cat = np.eye(10)[y_train]
    y_test_cat = np.eye(10)[y_test]

    # Build model
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
        ReLU(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3)),
        ReLU(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128),
        ReLU(),
        Dropout(0.5),
        Dense(10),
        Softmax()
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    print("Starting training (Subset)...")
    # Use ImageDataGenerator for normalization
    datagen = ImageDataGenerator(rescale=1/255.0)
    train_flow = datagen.flow(x_train[:1000], y_train_cat[:1000], batch_size=64)
    
    # Train for 1 epoch for demo purposes
    model.fit(train_flow, epochs=10, validation_data=(x_test[:100]/255.0, y_test_cat[:100]))

    print("Evaluating...")
    results = model.evaluate(x_test[:100]/255.0, y_test_cat[:100])
    print(f"Test Results: {results}")

if __name__ == "__main__":
    train_mnist()

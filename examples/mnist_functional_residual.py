import numpy as np
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neutro.models import Model
from neutro.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Add, ReLU, Softmax
from neutro.utils.data_utils import load_mnist
from neutro.preprocessing.image import ImageDataGenerator
from neutro.optimizers import Adam

def train_mnist_residual():
    """
    Trains a ResNet-style model on MNIST using the Functional API.
    This demonstrates non-linear connectivity (skip connections).
    """
    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Preprocess
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
    
    # One-hot encode labels
    y_train_cat = np.eye(10)[y_train]
    y_test_cat = np.eye(10)[y_test]

    # --- Build model using Functional API ---
    inputs = Input(shape=(28, 28, 1))
    
    # Initial block
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Residual Block
    # Shortcut path
    shortcut = x
    
    # Residual path
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    
    # Merge paths
    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    # Classification Head
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(10)(x)
    outputs = Softmax()(outputs)
    
    model = Model(inputs=inputs, outputs=outputs)
    # ----------------------------------------

    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    print("\nModel Summary:")
    model.summary()

    print("\nStarting training (Subset of 1000 samples for demo)...")
    datagen = ImageDataGenerator(rescale=1/255.0)
    train_flow = datagen.flow(x_train[:1000], y_train_cat[:1000], batch_size=64)
    
    # Train for 5 epochs for demo purposes
    model.fit(
        train_flow, 
        epochs=5, 
        validation_data=(x_test[:100]/255.0, y_test_cat[:100])
    )

    print("\nEvaluating on test set...")
    results = model.evaluate(x_test[:100]/255.0, y_test_cat[:100])
    print(f"Test Results: {results}")

if __name__ == "__main__":
    train_mnist_residual()

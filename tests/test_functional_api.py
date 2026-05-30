import numpy as np
import pytest
from neutro.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Add, Concatenate, GlobalAveragePooling2D
from neutro.models import Model
from neutro.optimizers import SGD

def test_linear_functional_model():
    """Test a simple linear stack built with functional API."""
    inputs = Input(shape=(10,))
    x = Dense(32, activation='relu')(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    X = np.random.randn(5, 10)
    y = model.predict(X)
    assert y.shape == (5, 1)

def test_skip_connection_model():
    """Test a model with a residual skip connection."""
    inputs = Input(shape=(32,))
    x = Dense(32, activation='relu')(inputs)
    residual = Dense(32, activation='relu')(x)
    merged = Add()([x, residual])
    outputs = Dense(10)(merged)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    X = np.random.randn(5, 32)
    y = model.predict(X)
    assert y.shape == (5, 10)
    
    # Test training and backprop through skip connection
    model.compile(optimizer=SGD(0.01), loss='mse')
    target = np.random.randn(5, 10)
    history = model.fit(X, target, epochs=1, verbose=0)
    assert 'loss' in history.history

def test_multi_input_model():
    """Test a model with two separate input branches."""
    input1 = Input(shape=(10,), name='input1')
    input2 = Input(shape=(20,), name='input2')
    
    x1 = Dense(16, activation='relu')(input1)
    x2 = Dense(16, activation='relu')(input2)
    
    merged = Concatenate()([x1, x2])
    outputs = Dense(1)(merged)
    
    model = Model(inputs=[input1, input2], outputs=outputs)
    
    X1 = np.random.randn(5, 10)
    X2 = np.random.randn(5, 20)
    y = model.predict([X1, X2])
    assert y.shape == (5, 1)

def test_multi_output_model():
    """Test a model with multiple outputs."""
    inputs = Input(shape=(10,))
    x = Dense(32, activation='relu')(inputs)
    
    output1 = Dense(1, name='out1')(x)
    output2 = Dense(5, name='out2')(x)
    
    model = Model(inputs=inputs, outputs=[output1, output2])
    
    X = np.random.randn(5, 10)
    y1, y2 = model.predict(X)
    assert y1.shape == (5, 1)
    assert y2.shape == (5, 5)

def test_conv_functional_model():
    """Test a convolutional model with functional API."""
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    outputs = Dense(10)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    X = np.random.randn(2, 28, 28, 1)
    y = model.predict(X)
    assert y.shape == (2, 10)

def test_functional_summary():
    """Test if summary() runs without error for functional models."""
    inputs = Input(shape=(10,))
    x = Dense(32)(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

def test_shared_layer():
    """Test using the same layer instance multiple times in a functional graph."""
    inputs = Input(shape=(10,))
    shared_dense = Dense(10, activation='relu')
    
    x1 = shared_dense(inputs)
    x2 = shared_dense(x1)
    
    model = Model(inputs=inputs, outputs=x2)
    
    X = np.random.randn(5, 10)
    y = model.predict(X)
    assert y.shape == (5, 10)
    
def test_functional_gradients():
    """Verify gradients in a functional model with a skip connection using finite differences."""
    inputs = Input(shape=(4,))
    x = Dense(8, activation='relu', kernel_initializer='ones', bias_initializer='zeros')(inputs)
    residual = Dense(8, activation='relu', kernel_initializer='ones', bias_initializer='zeros')(x)
    merged = Add()([x, residual])
    outputs = Dense(1, kernel_initializer='ones', bias_initializer='zeros')(merged)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=SGD(0.01), loss='mse')
    
    X = np.random.randn(1, 4)
    y_true = np.array([[1.0]])
    
    # Forward pass to cache values
    y_pred = model.forward(X, training=True)
    loss = model.loss_fn(y_true, y_pred)
    
    # Backward pass
    grad = model.loss_fn.gradient(y_true, y_pred)
    model.backward(grad)
    
    # Check gradient for one weight in a dense layer
    layer = model.layers[1] # First Dense layer
    W = layer.params['W']
    dW = layer.grads['W']
    
    eps = 1e-5
    i, j = 0, 0
    orig_val = W[i, j]
    
    W[i, j] = orig_val + eps
    y_plus = model.forward(X, training=False)
    loss_plus = model.loss_fn(y_true, y_plus)
    
    W[i, j] = orig_val - eps
    y_minus = model.forward(X, training=False)
    loss_minus = model.loss_fn(y_true, y_minus)
    
    W[i, j] = orig_val
    
    num_grad = (loss_plus - loss_minus) / (2 * eps)
    assert np.isclose(dW[i, j], num_grad, rtol=1e-4, atol=1e-5)
    
def test_complex_summary():
    """Test summary() for a multi-input, multi-output model."""
    i1 = Input(shape=(10,), name='input1')
    i2 = Input(shape=(10,), name='input2')
    merged = Add()([i1, i2])
    o1 = Dense(1, name='out1')(merged)
    o2 = Dense(1, name='out2')(merged)
    model = Model(inputs=[i1, i2], outputs=[o1, o2])
    model.summary()

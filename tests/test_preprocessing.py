import numpy as np
import pytest
from neutro.preprocessing.image import ImageDataGenerator

def test_image_data_generator_flow():
    x = np.random.randn(10, 32, 32, 3)
    y = np.random.randint(0, 10, (10,))
    
    # Disable random flips to check rescaling
    datagen = ImageDataGenerator(rescale=1/255.0, horizontal_flip=False)
    it = datagen.flow(x, y, batch_size=2)
    
    batch_x, batch_y = next(iter(it))
    assert batch_x.shape == (2, 32, 32, 3)
    assert batch_y.shape == (2,)
    # Check rescaling
    np.testing.assert_allclose(batch_x, x[it.indices[:2]] / 255.0, atol=1e-5)

def test_image_data_generator_transform():
    img = np.zeros((32, 32, 3))
    img[0, 0, 0] = 1.0 # Top-left pixel
    
    datagen = ImageDataGenerator(horizontal_flip=True)
    # Since it's random, we might need a few tries or set seed
    np.random.seed(42)
    
    # Check if flip works
    flipped = False
    for _ in range(10):
        transformed = datagen.apply_transform(img)
        if transformed[0, -1, 0] == 1.0:
            flipped = True
            break
    assert flipped

def test_image_data_generator_rotation():
    img = np.zeros((32, 32, 3))
    img[16, 16, :] = 1.0 # Center
    img[0, 16, :] = 1.0  # Top-center
    
    datagen = ImageDataGenerator(rotation_range=90)
    transformed = datagen.apply_transform(img)
    
    assert transformed.shape == (32, 32, 3)
    # Center should remain 1.0
    assert transformed[16, 16, 0] > 0.5
    # Top-center should move
    assert transformed[0, 16, 0] < 0.5

def test_image_data_generator_channels_first():
    x = np.random.randn(4, 3, 8, 8)
    y = np.random.randint(0, 10, (4,))

    datagen = ImageDataGenerator(rescale=1/255.0, data_format='channels_first', horizontal_flip=False)
    it = datagen.flow(x, y, batch_size=2, shuffle=False)

    batch_x, batch_y = next(iter(it))
    assert batch_x.shape == (2, 3, 8, 8)
    assert batch_y.shape == (2,)
    np.testing.assert_allclose(batch_x, x[:2] / 255.0, atol=1e-5)

def test_image_data_generator_invalid_data_format():
    with pytest.raises(ValueError, match="data_format must be"):
        ImageDataGenerator(data_format='invalid')

def test_image_data_generator_vertical_flip():
    img = np.zeros((32, 32, 3))
    img[0, :, 0] = 1.0

    datagen = ImageDataGenerator(vertical_flip=True)
    np.random.seed(42)

    flipped = False
    for _ in range(10):
        transformed = datagen.apply_transform(img)
        if np.all(transformed[-1, :, 0] == 1.0):
            flipped = True
            break
    assert flipped

def test_image_data_generator_width_height_shift():
    img = np.zeros((32, 32, 3))
    img[16, 16, :] = 1.0

    datagen = ImageDataGenerator(width_shift_range=0.5, height_shift_range=0.5)
    np.random.seed(0)
    transformed = datagen.apply_transform(img)

    assert transformed.shape == (32, 32, 3)

def test_image_data_generator_channels_first_vertical_flip():
    img = np.zeros((3, 32, 32))
    img[:, 0, :] = 1.0

    datagen = ImageDataGenerator(vertical_flip=True, data_format='channels_first')
    np.random.seed(42)

    flipped = False
    for _ in range(10):
        transformed = datagen.apply_transform(img)
        if np.all(transformed[:, -1, :] == 1.0):
            flipped = True
            break
    assert flipped

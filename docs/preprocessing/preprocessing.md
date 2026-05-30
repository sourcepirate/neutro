# Preprocessing

## Image Preprocessing — `neutro/preprocessing/image.py`

### ImageDataGenerator

Generates batches of image data with on-the-fly data augmentation:

```python
class ImageDataGenerator:
    def __init__(self, rotation_range=0, width_shift_range=0, height_shift_range=0,
                 horizontal_flip=False, zoom_range=0, **kwargs):
```

Supports:
- **Rotation**: Random rotation within `rotation_range` degrees.
- **Shift**: Random horizontal/vertical translation.
- **Flip**: Random horizontal flip.
- **Zoom**: Random zoom.
- **Normalization**: Rescaling, mean subtraction.

The generator implements `.flow(x, y, batch_size)` to yield batches:

```python
datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
for x_batch, y_batch in datagen.flow(X, y, batch_size=32):
    model.fit(x_batch, y_batch, epochs=1)
```

## Text Preprocessing — `neutro/preprocessing/text.py`

### Tokenizer utilities

Provides `text_to_word_sequence`, `one_hot`, `hashing_trick`, and other Keras-compatible text preprocessing functions:

```python
def text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
    # Split text into words
    ...
```

## Sequence Preprocessing — `neutro/preprocessing/sequence.py`

### pad_sequences

Pads sequences to the same length:

```python
def pad_sequences(sequences, maxlen=None, padding='pre', truncating='pre', value=0.0):
    # Pad or truncate to maxlen
    padded = np.full((len(sequences), maxlen), value)
    for i, seq in enumerate(sequences):
        ...
```

## References

- Keras Preprocessing API. [Keras.io](https://keras.io/api/preprocessing/)

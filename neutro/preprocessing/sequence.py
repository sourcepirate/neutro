import numpy as np


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre',
                  truncating='pre', value=0.0):
    """Pads sequences to the same length."""
    if padding not in ('pre', 'post'):
        raise ValueError(f'Padding type "{padding}" not understood')
    if truncating not in ('pre', 'post'):
        raise ValueError(f'Truncating type "{truncating}" not understood')

    if maxlen is None:
        maxlen = max((len(sequence) for sequence in sequences), default=0)

    result = np.full((len(sequences), maxlen), value, dtype=dtype)

    for index, sequence in enumerate(sequences):
        values = sequence[-maxlen:] if truncating == 'pre' and maxlen else sequence[:maxlen]
        if not len(values):
            continue

        values = np.asarray(values, dtype=dtype)
        if padding == 'post':
            result[index, :len(values)] = values
        else:
            result[index, -len(values):] = values

    return result

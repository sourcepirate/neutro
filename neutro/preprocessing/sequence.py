import numpy as np

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0):
    """
    Pads sequences to the same length.
    
    Args:
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
        padding: String, 'pre' or 'post': pad either before or after each sequence.
        truncating: String, 'pre' or 'post': remove values from sequences larger than
            maxlen, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
        
    Returns:
        x: Numpy array with shape `(len(sequences), maxlen)`
    """
    num_samples = len(sequences)
    
    lengths = [len(s) for s in sequences]
    if maxlen is None:
        maxlen = np.max(lengths)
        
    x = np.full((num_samples, maxlen), value, dtype=dtype)
    
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was passed
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')
            
        trunc = np.asarray(trunc, dtype=dtype)
        
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
            
    return x

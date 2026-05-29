import numpy as np

def get_im2col_indices(x_shape, field_height, field_width, padding=0, stride=1):
    # First figure out output sizes
    N, C, H, W = x_shape
    if isinstance(padding, int):
        ph, pw = padding, padding
    else:
        ph, pw = padding
        
    if isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride

    # assert (H + 2 * ph - field_height) % sh == 0
    # assert (W + 2 * pw - field_width) % sw == 0
    out_height = int((H + 2 * ph - field_height) // sh + 1)
    out_width = int((W + 2 * pw - field_width) // sw + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = sh * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = sw * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=0, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    if isinstance(padding, int):
        ph, pw = padding, padding
    else:
        ph, pw = padding
    # Zero-pad the input
    x_padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    if isinstance(padding, int):
        ph, pw = padding, padding
    else:
        ph, pw = padding
        
    H_padded, W_padded = H + 2 * ph, W + 2 * pw
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    if ph == 0 and pw == 0:
        return x_padded
    
    # Correct slicing for separate padding
    res = x_padded
    if ph > 0:
        res = res[:, :, ph:-ph, :]
    if pw > 0:
        res = res[:, :, :, pw:-pw]
    return res

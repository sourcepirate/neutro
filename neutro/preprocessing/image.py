import numpy as np
from scipy.ndimage import rotate, shift
from ..data import DataLoader

class ImageDataGenerator:
    """
    Generate batches of tensor image data with real-time data augmentation.
    
    Args:
        rotation_range: Degree range for random rotations.
        width_shift_range: Fraction of total width for random horizontal shifts.
        height_shift_range: Fraction of total height for random vertical shifts.
        horizontal_flip: Randomly flip inputs horizontally.
        vertical_flip: Randomly flip inputs vertically.
        rescale: Rescaling factor. If None or 0, no rescaling is applied, 
                 otherwise we multiply the data by the value provided.
        data_format: One of ``channels_last`` or ``channels_first``.
            ``channels_last`` expects images as ``(N, H, W, C)``,
            while ``channels_first`` expects ``(N, C, H, W)``.
            Internally transforms are applied in channels-last format and
            returned in the same format as provided.
    """
    def __init__(self, 
                 rotation_range=0, 
                 width_shift_range=0.0, 
                 height_shift_range=0.0, 
                 horizontal_flip=False, 
                 vertical_flip=False,
                 rescale=None,
                 data_format='channels_last'):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        if data_format not in ('channels_last', 'channels_first'):
            raise ValueError("data_format must be 'channels_last' or 'channels_first'")
        self.data_format = data_format

    def _to_channels_last(self, x):
        if self.data_format == 'channels_first':
            return np.transpose(x, (1, 2, 0))
        return x

    def _from_channels_last(self, x):
        if self.data_format == 'channels_first':
            return np.transpose(x, (2, 0, 1))
        return x

    def flow(self, x, y=None, batch_size=32, shuffle=True):
        """
        Takes data & label arrays, generates batches of augmented data.
        """
        return DataLoader(x, y, batch_size=batch_size, shuffle=shuffle, augmenter=self)

    def apply_transform(self, x):
        """
        Applies a transformation to an image.

        Args:
            x: Single image tensor in the configured layout.
                ``channels_last`` expects ``(H, W, C)``,
                ``channels_first`` expects ``(C, H, W)``.

        Returns:
            Augmented image tensor in the same layout as ``x``.
        """
        img = self._to_channels_last(x.copy().astype(np.float32))
        
        if self.rescale:
            img *= self.rescale

        if self.horizontal_flip and np.random.random() > 0.5:
            img = img[:, ::-1, :]
            
        if self.vertical_flip and np.random.random() > 0.5:
            img = img[::-1, :, :]

        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            # rotate along the first two axes (H, W)
            img = rotate(img, angle, axes=(0, 1), reshape=False, mode='nearest')

        if self.width_shift_range > 0 or self.height_shift_range > 0:
            h, w = img.shape[:2]
            tx = np.random.uniform(-self.width_shift_range, self.width_shift_range) * w
            ty = np.random.uniform(-self.height_shift_range, self.height_shift_range) * h
            img = shift(img, [ty, tx, 0], mode='nearest')

        return self._from_channels_last(img)

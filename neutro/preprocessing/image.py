import numpy as np
from scipy.ndimage import rotate, shift

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
    """
    def __init__(self, 
                 rotation_range=0, 
                 width_shift_range=0.0, 
                 height_shift_range=0.0, 
                 horizontal_flip=False, 
                 vertical_flip=False,
                 rescale=None):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale

    def flow(self, x, y=None, batch_size=32, shuffle=True):
        """
        Takes data & label arrays, generates batches of augmented data.
        """
        from ..data import DataLoader
        return DataLoader(x, y, batch_size=batch_size, shuffle=shuffle, augmenter=self)

    def apply_transform(self, x):
        """
        Applies a transformation to an image.
        x: (H, W, C)
        """
        img = x.copy().astype(np.float32)
        
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

        return img

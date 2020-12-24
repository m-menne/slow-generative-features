########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file contains the code for the generation of moving and static sequences with variation in position,
# identity, rotation and scaling.
########################################################################################################################

import numpy as np
from scipy.ndimage import zoom, rotate

# Fixing seeds for comparison
numpy_seed = 0
np.random.seed(numpy_seed)

class Item:
    """Class to store a given image with attributes.
    """
    def __init__(self, image):
        """Initialize an instance of the class with the given arguments.
        Args:
            image:  2D image to store in the item
        """
        self.image = image
        self.attributes = []

class SequenceGenerator:
    """Class to generate sequences based on the given input images.
    """
    def __init__(self, input, moving_sequence=True, output_dim=64):
        """Initialize an instance of the class with the given arguments.
        Args:
            input:              List of Item-objects
            moving_sequence:    Flag indicating whether the resulting sequence is a moving sequence
            output_dim:         Dimensionality of the output images
        """
        self.items = input
        self.moving_sequence = moving_sequence
        self.output_dim = output_dim
        
    def add_attribute(self, attributes):
        """Add a given attribute to the items.
        Args:
            attributes:     List of attributes to append to the stored items
        """
        if len(attributes) != len(self.items):
            raise RuntimeError("Number of given attribute values does not match number of current items.")
        
        for i, item in enumerate(self.items):
            item.attributes.append(attributes[i])
        
    def apply_rotation(self, angle_step=10):
        """Rotate the images.
        Args:
            angle_step:      Stepwidth of the rotation angle
        """
        input = self.items
        n_items = len(input)                        # Get number of input items
        n_rotations = 360 // angle_step             # Get number of rotations
        output = (n_items * n_rotations) * [None]   # Compute number of output items
        
        # Create output items
        idx = 0
        for item in input:
            for angle in range(n_rotations):
                new_item = Item(rotate(item.image, angle*angle_step, reshape=False)) # Apply rotation to image
                new_item.attributes = item.attributes[:]       # Copy old attribute values
                new_item.attributes.append(angle*angle_step)   # Append rotation attribute to item
                output[idx] = new_item
                idx += 1

        self.items = output

    def apply_scaling(self, n_scales=6, scale_step=0.2):
        """Scale the images.
        Args:
            n_scales:       Number of scalings
            scale_step:     Stepwidth of the scaling
        """
        if self.moving_sequence:
            raise RuntimeError("Scaling is not supported in moving sequence mode.")

        input = self.items
        n_item = len(input)                     # Get number of input items
        output = (n_item * n_scales) * [None]   # Compute number of output items

        # Create output items
        idx = 0
        for item in input:
            for scale in range(n_scales):
                new_item = Item(zoom(item.image, 1 + scale * scale_step))    # Apply scaling to image
                new_item.attributes = item.attributes[:]            # Copy old attribute values
                new_item.attributes.append(1 + scale * scale_step)  # Append scaling attribute to item
                output[idx] = new_item
                idx += 1
                
        self.items = output
        
    def apply_translation(self, n_positions=18, pos_step=2):
        """Translate sub-image in sequence. Note that the translation is only after padding in the images visible.
        Args:
            n_positions:    Number of positions per dimension
            pos_step:       Stepwidth between each position
        """
        if not self.moving_sequence:
            raise RuntimeError("Translation only available in moving sequence mode.")
        input = self.items
        n_item = len(input)                                         # Get number of input items
        output = (n_item * n_positions * n_positions) * [None]      # Compute number of output items
        
        # Create output items
        idx = 0
        for item in input:
            for x in range(n_positions):
                for y in range(n_positions):
                    new_item = Item(item.image)
                    new_item.attributes = item.attributes[:]                            # Copy old attribute values
                    new_item.attributes.append((x * pos_step, y * pos_step))    # Store position attributes
                    output[idx] = new_item
                    idx += 1
                    
        self.items = output
    
    def _pad_images(self):
        """Pad sub-images into output images.
        """
        if self.moving_sequence:    # Pad with respect to the position of the sub-image in the case of a moving sequence
            if np.array(self.items[0].attributes[-1]).ndim != 1:
                raise RuntimeError("Translation has to be performed as the last operation.")
            diff_shape = self.output_dim - np.array(self.items[0].image).shape[0]
            for item in self.items:
                (x_pos, y_pos) = item.attributes[-1]
                pad_left = np.array([x_pos, y_pos])
                pad_right = [diff_shape, diff_shape] - pad_left
                item.image = np.expand_dims(np.pad(item.image, ((pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]))), axis=2)
        else:                       # Pad the sub-images into the center of the output images
            for item in self.items:
                diff_shape = self.output_dim - np.array(item.image).shape[0]
                pad_left = diff_shape//2
                pad_right = diff_shape - pad_left
                item.image = np.expand_dims(np.pad(item.image, ((pad_left,pad_right), (pad_left,pad_right))), axis=2)
    
    
    def generateData(self, rescaling=True, shuffle=False):
        """Generate, rescale and shuffle the items.
        Args:
            rescaling:      Flag indicating whether images should be rescaled to [0.,1.] interval
            shuffle:        Flag indicating whether the dataset should be shuffled
        Return:
            List of output items including the final images and attributes
        """
        # Pad images to match the set output dimensionality
        self._pad_images()
        
        # Normalize the images to the range of [0., 1.]
        if rescaling:
            for item in self.items:
                item.image = item.image / 255.
        
        # Shuffle the items
        if shuffle:
            np.random.shuffle(self.items)

        return self.items

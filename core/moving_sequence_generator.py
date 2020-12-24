########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file contains the code for the generation of moving sequences.
# The code is adapted from the Tensorflow.Datasets API (accessed on 9 March 2020):
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/video/moving_sequence.py
########################################################################################################################

import numpy as np
import time

from datetime import datetime
def log(message):
    """Function to log messages with current time to the terminal.
    """
    print("[" + datetime.now().strftime("%H:%M:%S") + "][MovingSequenceGenerator] " + message)
    
# Fixing seed for comparison
numpy_seed = 0
np.random.seed(numpy_seed)

class MovingSequenceGenerator():
    """Class to generate moving sequences based on the given input images.
    """
    
    def __init__(self, image_data, n_seq, seq_length, dim_out_img=64, velocity=0.1, start_position=None, connected_seq=False, variation_mode=False, stochastic_mode=False):
        """Initialize an instance of the class with the given arguments.
        Args:
            image_data:       3-dimensional array [n_images, img_h, img_w] or 4-dimensional array [n_images, n_variations, img_h, img_w] in variation mode containing the input images
            n_seq:            Total number of sequences to generate (integer)
            seq_length:       Length of each sequence (integer)
            dim_out_img:      Dimensionality of the output images (integer)
            velocity:         Scalar speed or 2D velocity of input images
            start_position:   Starting position of the input images within the sequences
            connected_seq:    Flag indicating whether the sequences should be connected to one long trajectory
            variation_mode:   Variation mode allows the variation of input images within a sequence
            stochastic_mode:  Stochastic mode generates trajectories with stochastic reflections at the edges as introduced in Stochastic Moving MNIST (Denton and Fergus, 2018)
        """
        self.image_data = np.array(image_data)
        if variation_mode and self.image_data.ndim is not 4:
            raise RuntimeError("The given input images should be passed as a 4-dimensional array [n_images, n_variations, img_h, img_w] in variation mode.")
        elif not variation_mode and self.image_data.ndim is not 3:
            raise RuntimeError("The given input images should be passed as a 3-dimensional array [n_images, img_h, img_w].")
        self.n_seq = n_seq
        self.seq_length = seq_length
        self.dim_img = dim_out_img
        self.velocity = velocity
        self.start_pos = start_position
        self.connected_seq = connected_seq
        self.variation_mode = variation_mode
        self.stochastic_mode = stochastic_mode
        self.data = None

    def _get_random_unit_vector(self, ndims=2, dtype=np.float32):
        x = np.random.normal(size=(ndims,))
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
        
    def _get_linear_trajectory(self, start_position, velocity, t):
        """Construct a linear trajectory starting from start_position.
        Args:
            start_position: Start position of the trajectory
            velocity:       Array containing the velocity
            t:              [seq_length]-length float array
        Returns:
            x:              [seq_length, ndims] float array containing the trajectory points
        """
        start_position = np.expand_dims(start_position, axis=0)
        velocity = np.expand_dims(velocity, axis=0)
        dx = velocity * np.expand_dims(t, axis=-1)
        return start_position + dx

    def _bounce_to_bbox(self, trajectory):
        """Bounce potentially unbounded points to [0, 1].
        Bouncing occurs by exact reflection, i.e. a pre-bound point at 1.1 is moved
        to 0.9, -0.2 -> 0.2. This theoretically can occur multiple times, e.g. 2.3 -> -0.7 -> 0.3
        Args:
            trajectory: Float array containing the trajectory points
        Returns:
            Array with same shape/dtype but values in [0, 1].
        """
        trajectory = trajectory % 2
        return np.minimum(2-trajectory, trajectory)
        
    def _bounce_to_bbox_stochastic(self, trajectory):
        """Bounce potentially unbounded points to [0, 1].
        Bouncing occurs stochastically by randomly reflecting and choosing random velocities after each reflection.
        Args:
            trajectory: Float array containing the trajectory points
        Returns:
            Array with the same shape/dtype but values in [0, 1].
        """
        # Check if each position of the trajectory is inside the interval [0, 1]
        # If not adjust the trajectory such that the sub-images are reflected randomly from the boarder
        ndims = 2
        for i in range(self.seq_length):
            while trajectory[i][0] < 0 or trajectory[i][0] > 1 or trajectory[i][1] < 0 or trajectory[i][1] > 1: # Check if one constraint is not fulfilled
                if trajectory[i][0] < 0:    # Collides with left boarder
                        velo_alt = np.random.uniform(0.025,0.1)                                             # Compute new random velocity
                        t_alt = np.arange(self.seq_length-i+1)                                              # Compute basis for new trajectory
                        dir = self._get_random_unit_vector(ndims, np.float32)                               # Compute new direction vector
                        velocity =  np.array([np.abs(dir[0]), dir[1]]) * velo_alt                           # Adjust direction so that it points inside the image
                        trajectory_new = self._get_linear_trajectory(trajectory[i-1], velocity, t_alt)      # Compute the new trajectory starting from the last valid point of the trajectory
                        trajectory[i:] = trajectory_new[1:]                                                 # Override old invalid trajectory with new trajectory
                
                elif trajectory[i][0] > 1:  # Collides with right boarder
                        velo_alt = np.random.uniform(0.025,0.1)
                        t_alt = np.arange(self.seq_length-i+1)
                        dir = self._get_random_unit_vector(ndims, np.float32)
                        velocity =  np.array([-np.abs(dir[0]), dir[1]]) * velo_alt
                        trajectory_new = self._get_linear_trajectory(trajectory[i-1], velocity, t_alt)
                        trajectory[i:] = trajectory_new[1:]

                elif trajectory[i][1] < 0:  # Collides with lower boarder
                        velo_alt = np.random.uniform(0.025,0.1)
                        t_alt = np.arange(self.seq_length-i+1)
                        dir = self._get_random_unit_vector(ndims, np.float32)
                        velocity =  np.array([dir[0], np.abs(dir[1])]) * velo_alt
                        trajectory_new = self._get_linear_trajectory(trajectory[i-1], velocity, t_alt)
                        trajectory[i:] = trajectory_new[1:]

                elif trajectory[i][1] > 1:  # Collides with upper boarder
                        velo_alt = np.random.uniform(0.025,0.1)
                        t_alt = np.arange(self.seq_length-i+1)
                        dir = self._get_random_unit_vector(ndims, np.float32)
                        velocity =  np.array([dir[0], -np.abs(dir[1])]) * velo_alt
                        trajectory_new = self._get_linear_trajectory(trajectory[i-1], velocity, t_alt)
                        trajectory[i:] = trajectory_new[1:]
                        
        return trajectory
        
    def _create_moving_sequence(self, image, pad_lefts, total_padding):
        """Create a moving image sequence from the given image/images and left padding values.
        Args:
            image:          [in_h, in_w] array or [seq_length, in_h, in_w] array in variation mode
            pad_lefts:      [seq_length, 2] array of left padding values
            total_padding:  array of padding values (pad_h, pad_w)
        Returns:
            [seq_length, out_h, out_w] image sequence, where out_h = in_h + pad_h, out_w = in_w + out_w
        """
        def get_padded_image(image, pad_left):
            pad_right = total_padding - pad_left
            padding = np.stack([pad_left, pad_right], axis=-1)
            return np.pad(image, padding)
            
        if self.variation_mode:
            if pad_lefts.shape[0] != image.shape[0]:
                raise RuntimeError("Number of passed sub-images used as variation has to be equal to sequence length in variation mode.")
            padded_images = [get_padded_image(image[idx], pad_lefts[idx]) for idx in range(pad_lefts.shape[0])]
        else:
            padded_images = [get_padded_image(image, pad_lefts[idx]) for idx in range(pad_lefts.shape[0])]
                
        return padded_images
    
    def image_as_moving_sequence(self, image, seq_length=20, output_size=(64,64), velocity=0.1, start_position=None, seq_idx=None):
        """Turn simple static images into sequences of the originals bouncing around.
        Adapted from Srivastava et al.: http://www.cs.toronto.edu/~nitish/unsupervised_video/

        Args:
            image:          [in_h, in_w] tensor defining the sub-image to be bouncing around.
                            In variation mode, not a single sub-image but a set of sub-images is passed, which are used as the images for the sequence.
            seq_length:     Length of sequence (integer)
            output_size:    (out_h, out_w) size of returned images
            velocity:       Scalar speed or 2D velocity of image. If scalar, the 2D velocity is randomly generated with this magnitude.
                            This is the normalized distance moved each time step by the sub-image, where normalization occurs over the
                            feasible distance the sub-image can move e.g if the input image is [10 x 10] and the output image is [60 x 60],
                            a speed of 0.1 means the sub-image moves (60 - 10) * 0.1 = 5 pixels per time step.
            start_position: 2D float32 normalized initial position of each image in [0, 1]. Randomized uniformly if not given.
            seq_idx:        Index of the sequence. Keep in mind that the dimension of the output is increased by two entries.
        Returns:
            [sequence_length, output_size[0]*output_size[1]+2+2] tensor containing the images with appended position and sequence indices
        """
        ndims = 2
        image = np.array(image)
        output_size = np.array(output_size)
        if start_position is None:
            start_position = np.random.uniform(size=(ndims,))
        
        velocity = np.array(velocity)
        if velocity.shape == ():
            if self.stochastic_mode: # If stochastic mode, choose velocity for each trajectory randomly
                velocity = np.random.uniform(0.025,0.1)
            velocity = self._get_random_unit_vector(ndims, np.float32) * velocity
        
        t = np.arange(self.seq_length)
        trajectory = self._get_linear_trajectory(start_position, velocity, t)
            
        if self.stochastic_mode:
            trajectory = self._bounce_to_bbox_stochastic(trajectory)    # Compute stochastic trajectory for Stochastic Moving MNIST
        else:
            trajectory = self._bounce_to_bbox(trajectory)               # Compute deterministic trajectory for Moving MNIST
        
        total_padding = output_size - image.shape[-2:]
        sequence_pad_lefts = np.around(trajectory * total_padding.astype(np.float32)).astype(np.int32)
        
        sequence = self._create_moving_sequence(image, sequence_pad_lefts, total_padding)
        
        # Create indices if sequence index is given
        if seq_idx is not None:
            indices = [[seq_idx, idx] for idx in range(seq_length)]
            
        sequence = np.reshape(sequence, [seq_length, output_size[0]*output_size[1]])
        sequence = np.concatenate([sequence, trajectory], axis=1)
        if seq_idx is not None:
            sequence = np.concatenate([sequence, indices], axis=1)
        
        return sequence

    def generateData(self):
        """Generates an array with the generated sequences of the input images bouncing around.
        Return:
          self.data: [self.image_data.shape[0] * sequence_length, output_size[0]*output_size[1]+2+2] array containing the images with appended position and sequence indices
        """
        log("Generating moving sequence data.")
        
        # Duplicate input images to match the desired number of sequences to generate
        n_diff_images = self.image_data.shape[0]
        if self.n_seq%n_diff_images is not 0:
            raise RuntimeError("The desired number of sequences (n_seq) cannot be generated with the given number of input images and set option parameters.")
        n_dup = self.n_seq // n_diff_images
        self.image_data = np.array([self.image_data for _ in range(n_dup)])
        
        if self.variation_mode:
            self.image_data = np.reshape(self.image_data, (self.n_seq, self.seq_length, self.image_data.shape[-2], self.image_data.shape[-1]))
        else:
            self.image_data = np.reshape(self.image_data, (self.n_seq, self.image_data.shape[-2], self.image_data.shape[-1]))
        
        # Generate the data
        size = self.image_data.shape[0] * self.seq_length
        data = size * [None]
         
        if self.connected_seq:
            last_pos = self.start_pos
            for idx in range(self.image_data.shape[0]):
                sequence = self.image_as_moving_sequence(self.image_data[idx], seq_length=self.seq_length, output_size=(self.dim_img, self.dim_img), velocity=self.velocity, start_position=last_pos, seq_idx=idx)
                last_pos = sequence[-1,-4:-2]
                write_off = idx * self.seq_length
                for i, img in enumerate(sequence):
                    data[write_off + i] = img
                    
        else:
            for idx in range(self.image_data.shape[0]):
                sequence = self.image_as_moving_sequence(self.image_data[idx], seq_length=self.seq_length, output_size=(self.dim_img, self.dim_img), velocity=self.velocity, start_position=self.start_pos, seq_idx=idx)
                write_off = idx * self.seq_length
                for i, img in enumerate(sequence):
                    data[write_off + i] = img
                    
        self.data = np.array(data)
            
        return self.data
        
    def generateDataset(self, data_split=[0.6, 0.2], normalization=False, rescaling=True, shuffle=True):
        """Postprocessing the generated data including splitting, normalizing, rescaling and shuffling.
        Args:
            data_split:         Float array containing the ratio of training and validation data
            normalization:      Flag indicating whether images should be normalized to zero mean and std. dev. one
            rescaling:          Flag indicating whether images should be rescaled to [0.,1.] interval
            shuffle:            Flag indicating whether the dataset should be shuffled
        Return:
            train_images:       [n_train_images, img_h, img_w] array of training images
            train_positions:    2D array containing the positions of the sub-images
            train_indices:      2D array containing the sequence number to which the image belongs and the index of the image within the sequence
            train_similarity:   2D array containing the similarity matrix of the training data based on temporal similarity
            val_images:         [n_val_images, img_h, img_w] array of validation images
            val_positions:      2D array containing the positions of the sub-images
            val_indices:        2D array containing the sequence number to which the image belongs and the index of the image within the sequence
            val_similarity:     2D array containing the similarity matrix of the validation data based on temporal similarity
            test_images:        [n_test_images, img_h, img_w] array of test images
            test_positions:     2D array containing the positions of the sub-images
            test_indices:       2D array containing the sequence number to which the image belongs and the index of the image within the sequence
            test_similarity:    2D array containing the similarity matrix of the test data based on temporal similarity
        """
        if self.data is None:
            self.generateData()
        
        dataset = self.data
        
        # Calculate total number of images in dataset
        n_images = self.n_seq * self.seq_length
        
        # Calculate number of training and validation sequences and images
        n_train_seq = int(data_split[0] * self.n_seq)
        n_train_images = n_train_seq * self.seq_length
        
        n_val_seq = int(data_split[1] * self.n_seq)
        n_val_images = n_val_seq * self.seq_length
        
        # Create mapping array which is essentially used when dataset will be shuffled
        mapping = np.arange(n_images)
        
        log("Splitting, normalizing, rescaling and shuffling the moving sequence data.")
        # Shuffle dataset
        if shuffle:
            np.random.shuffle(mapping)
            # Rearange positions of images in the dataset by using the random mapping
            dataset_dummy = n_images * [None]
            for i in range(n_images):
                dataset_dummy[mapping[i]] = dataset[i]
            
            self.data = dataset = np.array(dataset_dummy)
            
        # Split training, validation and test data into images, positions and indices
        train_images = np.reshape(dataset[:n_train_images,:(self.dim_img*self.dim_img)], [n_train_images, self.dim_img, self.dim_img,1])
        train_positions = dataset[:n_train_images,(self.dim_img*self.dim_img):-2]
        train_indices = dataset[:n_train_images,-2:]

        val_images = np.reshape(dataset[n_train_images:n_train_images+n_val_images,:(self.dim_img*self.dim_img)], [n_val_images, self.dim_img, self.dim_img,1])
        val_positions = dataset[n_train_images:n_train_images+n_val_images,(self.dim_img*self.dim_img):-2]
        val_indices = dataset[n_train_images:n_train_images+n_val_images,-2:]

        test_images = np.reshape(dataset[n_train_images+n_val_images:,:(self.dim_img*self.dim_img)], [(self.n_seq*self.seq_length) - (n_train_images+n_val_images), self.dim_img, self.dim_img,1])
        test_positions = dataset[n_train_images+n_val_images:,(self.dim_img*self.dim_img):-2]
        test_indices = dataset[n_train_images+n_val_images:,-2:]
        
        # Normalize training, validation and test images to zero mean and one std. dev.
        if normalization:
            train_images = train_images - np.mean(train_images, axis=0)
            std = np.std(train_images, axis=0)
            std[std == 0] = 1.
            train_images = train_images / std
            
            val_images = val_images - np.mean(val_images, axis=0)
            std = np.std(val_images, axis=0)
            std[std == 0] = 1.
            val_images = val_images / std
            
            test_images = test_images - np.mean(test_images, axis=0)
            std = np.std(test_images, axis=0)
            std[std == 0] = 1.
            test_images = test_images / std
            
        # Normalize the images to the range of [0., 1.]
        if rescaling:
            train_images = train_images / 255.
            val_images = val_images / 255.
            test_images = test_images / 255.
        
        # Generating similarity matrices based on temporal similarity
        log("Creating similiarity matrices.")
        start_time = time.time()
        
        # Initialize adjacency list
        adj_list = [[] for _ in range(n_images)]

        # Compute adjacency list in the connected sequence case
        if self.connected_seq:
            adj_list[mapping[0]].append(mapping[1])
            adj_list[mapping[-1]].append(mapping[-2])
            for i in range(n_images-2):
                adj_list[mapping[i+1]].append(mapping[i])
                adj_list[mapping[i+1]].append(mapping[i+2])
        
        # Compute adjacency list in the case where the sequences are not connected
        else:
            adj_list[mapping[0]].append(mapping[1])
            adj_list[mapping[-1]].append(mapping[-2])
            for i in range(n_images-2):
                if (i+1+1)%self.seq_length is 0:    # If sequence ends, only adjacend to previous image
                    adj_list[mapping[i+1]].append(mapping[i])
                elif (i+1)%self.seq_length is 0:    # If sequence starts, only adjacend to following image
                    adj_list[mapping[i+1]].append(mapping[i+2])
                else:                               # Else adjacend to previous and following image
                    adj_list[mapping[i+1]].append(mapping[i])
                    adj_list[mapping[i+1]].append(mapping[i+2])
            
        # Compute adjacency matrix based on adjacence list
        similarity = np.eye(n_images, dtype=np.int8)
        for i in range(n_images):
            for j in range(len(adj_list[i])):
                similarity[i, adj_list[i][j]] = 1
                
        # Split similarity matrix into train, validation and test similarity matrices
        train_similarity = similarity[:train_images.shape[0], :train_images.shape[0]]
        val_similarity = similarity[train_images.shape[0]:train_images.shape[0]+val_images.shape[0], train_images.shape[0]:train_images.shape[0]+val_images.shape[0]]
        test_similarity = similarity[train_images.shape[0]+val_images.shape[0]:, train_images.shape[0]+val_images.shape[0]:]
        
        log("Created similarity matrices in " + str(time.time()-start_time) + " seconds.")
         
        return train_images, train_positions, train_indices, train_similarity, val_images, val_positions, val_indices, val_similarity, test_images, test_positions, test_indices, test_similarity
        
    def apply_rotation(self, angle_step):
        """Apply rotation to input images from sequence to sequence.
        Args:
            angle_step:     Increasing rotation angle per sequence
        """
        if self.data is not None:
            print("Warning: This method should be applied before the moving sequence data is generated!")
            import sys
            sys.exit()
        if self.variation_mode:
            raise RuntimeError("The inclusion of rotation is currently not supported in variation mode.")
            
        n_diff_images = self.image_data.shape[0]
        n_rotations = 360 // angle_step
    
        # Duplicate images in preparation for rotation
        images = [img for img in self.image_data for _ in range(n_rotations)]
        images = np.reshape(images, (n_diff_images*n_rotations, self.image_data.shape[-2], self.image_data.shape[-1]))
        
        # Apply rotation to images
        from scipy.ndimage import rotate
        rotation_angle = 0
        for i in range(images.shape[0]):
            images[i] = rotate(images[i], rotation_angle, reshape=False)
            rotation_angle = (rotation_angle + angle_step) % 360
            
        self.image_data = images
        
    def animate_sequence(self, sequence_idx=0):
        """Animate one sequence contained in the data.
        Args:
            sequence_idx: Index of the sequence that should be animated
        """
        if sequence_idx > self.n_seq:
            raise RuntimeError("Sequence number out of bound.")
    
        if self.data is None:
            print("Warning: Automatic generation of dataset. This could take some time!")
            self.generateData()
            
        # Get sequence
        start_pos_seq = sequence_idx * self.seq_length
        end_pos_seq = start_pos_seq + self.seq_length
        sequence = np.reshape(self.data[start_pos_seq:end_pos_seq,:(self.dim_img*self.dim_img)], [self.seq_length, self.dim_img, self.dim_img])
        
        # Animate sequence
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig = plt.figure()
        plt.axis("off")
        ims = [[plt.imshow(im, cmap="gray", animated=True)] for im in sequence]
        anim = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
        plt.show()
        plt.close()
        
    def animate_multiple_sequences(self, n_seq=5):
        """Animate first sequences contained in the data.
        Args:
            n_seq: Number of sequences that should be animated
        """
        if self.data is None:
            print("Warning: Automatic generation of dataset. This could take some time!")
            self.generateData()
            
        if self.n_seq < n_seq:
            n_seq = self.n_seq
            print("WARNING: Dataset only contains " + str(n_seq) + " sequences.")
            
        # Get sequences
        start_pos_seq = 0
        end_pos_seq = start_pos_seq + self.seq_length * n_seq
        sequence = np.reshape(self.data[start_pos_seq:end_pos_seq,:(self.dim_img*self.dim_img)], [self.seq_length*n_seq, self.dim_img, self.dim_img])
        
        # Animate sequences
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig = plt.figure()
        plt.axis("off")
        ims = [[plt.imshow(im, cmap="gray", animated=True)] for im in sequence]
        anim = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=100)
        plt.show()
        plt.close()

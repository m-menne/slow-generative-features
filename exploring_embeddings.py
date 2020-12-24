########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file contains the code for the investigation of the embeddings in Section 3.4.1 & 3.4.2 of the paper.
# It provides the Slowness-Regularized Autoencoder model and the generation of moving as well as static
# sequence datasets with variation in position, identity, rotation and scaling.
########################################################################################################################

# Disable AVX2 FMA warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Disable TensorFlow 1.x "deprecated" warning messages
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from datetime import datetime
def log(message):
    """Function to log messages with current time to the terminal.
    """
    print("[" + datetime.now().strftime("%H:%M:%S") + "][Exploring embeddings] " + message)
    
import time
execution_time = time.time()

from keras.datasets import mnist

import numpy as np

from core.utils import *
from core.sequence_generator import Item, SequenceGenerator
from models import SRAE
from core.powersfa import ordered_gsfa_loss, GeneralizedBatchGenerator2
from core.plotter import Plotter

# Fixing seeds for comparison
numpy_seed = 0
np.random.seed(numpy_seed)

# Verification of the directory structure required for the storage of the results
check_dir_structure()

######################################### Setup ########################################################################
# Sequence Generation
dataset = 'rotation-scaling'            # Select dataset: rotation-scaling, rotation-identity, position-rotation, position-identity

# Model Configuration
n_features = 3                          # Dimensionality of the latent space
encoder_loss = ordered_gsfa_loss        # Loss function used to train the encoder
decoder_loss = 'binary_crossentropy'    # Loss function used to train the decoder
optimizer = 'nadam'                     # Optimizer used for training

# Training Parameters
batch_size = 1/5                        # Fraction of images that form a batch
n_epochs = 500                          # Number of epochs to train the model
train_model = False                     # Train the model (True) or load model weights (False)
save_weights = False                    # Save model weights

visualize_embedding = True              # Visualize the computed embedding in a 3D-plot

######################################### Dataset generation ###########################################################
log("Generating " + dataset + " dataset.")
# Loading MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if dataset is 'rotation-scaling':
    # Dataset configuration
    moving_sequence = False     # Flag indicating whether the resulting sequence is a moving sequence
    angle_step = 10             # Stepwidth of the rotation angle
    n_scales = 6                # Number of scalings
    scale_step = 0.2            # Stepwidth of the scaling
    shuffle = False             # Shuffle the items
    
    # Build the dataset
    seq_gen = SequenceGenerator([Item(x_train[2])], moving_sequence)    # Initialize the SequenceGenerator
    seq_gen.apply_rotation(angle_step)                                  # Apply rotation
    seq_gen.apply_scaling(n_scales, scale_step)                         # Apply scaling
    train_items = seq_gen.generateData(shuffle=shuffle)                 # Generate the data
    
    # Compute the similarity matrix
    n_items = len(train_items)
    W = np.zeros((n_items, n_items))
    
    for i in range(n_items):
        for j in range(n_items):
            if abs(train_items[i].attributes[1] - train_items[j].attributes[1]) < 2*scale_step and train_items[i].attributes[0] == train_items[j].attributes[0]:
                W[i,j] = W[j,i] = 1
            elif (abs(train_items[i].attributes[0] - train_items[j].attributes[0]) <= 2*angle_step or abs(train_items[i].attributes[0] - train_items[j].attributes[0]) == (360-angle_step)) and train_items[i].attributes[1] == train_items[j].attributes[1]:
                W[i,j] = W[j,i] = 1
    
elif dataset is 'rotation-identity':
    # Dataset configuration
    moving_sequence = False     # Flag indicating whether the resulting sequence is a moving sequence
    angle_step = 10             # Stepwidth of the rotation angle
    n_ids = 10                  # Number of different identities
    shuffle = False             # Shuffle the items
    
    # Get first image in the MNIST dataset of every used identity
    used_ids = np.arange(n_ids)
    images = [[x_train[i] for i in range(x_train.shape[0]) if y_train[i] == used_ids[j]] for j in range(used_ids.shape[0])]
    images = np.array([img[0] for img in images])
    items = [Item(img) for img in images]
    
    # Build the dataset
    seq_gen = SequenceGenerator(items, moving_sequence)     # Initialize the SequenceGenerator
    seq_gen.add_attribute(np.arange(n_ids))                 # Add corresponding identity-labels to items
    seq_gen.apply_rotation(angle_step)                      # Apply rotation
    train_items = seq_gen.generateData(shuffle=shuffle)     # Generate the data
    
    # Compute the similarity matrix
    n_items = len(train_items)
    W = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(n_items):
            if abs(train_items[i].attributes[0] - train_items[j].attributes[0]) < 2 and train_items[i].attributes[1] == train_items[j].attributes[1]:
                W[i,j] = W[j,i] = 1
            elif (abs(train_items[i].attributes[1] - train_items[j].attributes[1]) <= 2*angle_step or abs(train_items[i].attributes[1] - train_items[j].attributes[1]) == (360-angle_step)) and train_items[i].attributes[0] == train_items[j].attributes[0]:
                W[i,j] = W[j,i] = 1
    
elif dataset is 'position-rotation':
    # Dataset configuration
    moving_sequence = True      # Flag indicating whether the resulting sequence is a moving sequence
    angle_step = 20             # Stepwidth of the rotation angle
    n_positions = 18            # Number of positions per dimension
    pos_step = 2                # Stepwidth between each position
    shuffle = False             # Shuffle the items
    
    # Build the dataset
    seq_gen = SequenceGenerator([Item(x_train[2])], moving_sequence)    # Initialize the SequenceGenerator
    seq_gen.apply_rotation(angle_step)                                  # Apply rotation
    seq_gen.apply_translation(n_positions, pos_step)                    # Apply translation
    train_items = seq_gen.generateData(shuffle=shuffle)                 # Generate the data
    
    # Compute the similarity matrix
    n_items = len(train_items)
    W = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(n_items):
            if (abs(train_items[i].attributes[1][0] - train_items[j].attributes[1][0]) <= 2*pos_step and abs(train_items[i].attributes[1][1] - train_items[j].attributes[1][1]) <= 2*pos_step) and train_items[i].attributes[0] == train_items[j].attributes[0]:
                W[i,j] = W[j,i] = 1
            elif (abs(train_items[i].attributes[0] - train_items[j].attributes[0]) <= 2*angle_step or abs(train_items[i].attributes[0] - train_items[j].attributes[0]) == (360-angle_step)) and (train_items[i].attributes[1][0] == train_items[j].attributes[1][0] and train_items[i].attributes[1][1] == train_items[j].attributes[1][1]):
                W[i,j] = W[j,i] = 1
    
elif dataset is 'position-identity':
    # Dataset configuration
    moving_sequence = True      # Flag indicating whether the resulting sequence is a moving sequence
    n_ids = 10                  # Number of different identities
    n_positions = 18            # Number of positions per dimension
    pos_step = 2                # Stepwidth between each position
    shuffle = False             # Shuffle the items
    
    # Get first image in the MNIST dataset of every used identity
    used_ids = np.arange(n_ids)
    images = [[x_train[i] for i in range(x_train.shape[0]) if y_train[i] == used_ids[j]] for j in range(used_ids.shape[0])]
    images = np.array([img[0] for img in images])
    items = [Item(img) for img in images]
    
    # Build the dataset
    seq_gen = SequenceGenerator(items, moving_sequence)     # Initialize the SequenceGenerator
    seq_gen.add_attribute(np.arange(n_ids))                 # Add corresponding identity-labels to items
    seq_gen.apply_translation(n_positions, pos_step)        # Apply translation
    train_items = seq_gen.generateData(shuffle=shuffle)     # Generate the data
    
    # Compute the similarity matrix
    n_items = len(train_items)
    W = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(n_items):
            if (abs(train_items[i].attributes[1][0] - train_items[j].attributes[1][0]) <= 2*pos_step and abs(train_items[i].attributes[1][1] - train_items[j].attributes[1][1]) <= 2*pos_step) and train_items[i].attributes[0] == train_items[j].attributes[0]:
                W[i,j] = W[j,i] = 1
            elif abs(train_items[i].attributes[0] - train_items[j].attributes[0]) < 2 and (train_items[i].attributes[1][0] == train_items[j].attributes[1][0] and train_items[i].attributes[1][1] == train_items[j].attributes[1][1]):
                W[i,j] = W[j,i] = 1
    
else:
    raise RuntimeError("The selected dataset does not exist.")

train_images = val_images = np.array([item.image for item in train_items])
train_similarity = val_similarity = W

######################################### Slowness-Regularized Autoencoder model #######################################

log("Configuring the SRAE model.")
srae = SRAE(n_features).model
srae.compile(loss=[encoder_loss, decoder_loss], loss_weights=[15., 1.], optimizer=optimizer)

print(srae.summary())

# Train the SRAE model
if train_model:
    # Prepare a batch generator for the training and validation data
    train_gen = GeneralizedBatchGenerator2(train_images, train_similarity/int(train_images.shape[0]*batch_size), int(train_images.shape[0]*batch_size))
    val_gen = GeneralizedBatchGenerator2(val_images, val_similarity/int(train_images.shape[0]*batch_size), int(train_images.shape[0]*batch_size))

    log("Start training the SRAE model.")
    history_srae = srae.fit_generator(train_gen, int(1/batch_size), epochs=n_epochs, validation_data=val_gen, validation_steps=len(val_gen))
    if save_weights:
        log("Saving trained SRAE model weights.")
        srae.save_weights('Run/srae_weights.h5')

else:   # Alternatively load SRAE model weights
    log("Loading stored SRAE model weights.")
    if dataset is 'rotation-scaling':
        srae.load_weights('pretrained_models/exploring_embeddings/srae_weights_rot_scal.h5')
    elif dataset is 'rotation-identity':
        srae.load_weights('pretrained_models/exploring_embeddings/srae_weights_rot_id.h5')
    elif dataset is 'position-rotation':
        srae.load_weights('pretrained_models/exploring_embeddings/srae_weights_pos_rot.h5')
    else:
        srae.load_weights('pretrained_models/exploring_embeddings/srae_weights_pos_id.h5')

log("Computing predictions.")
[train_embedding_srae, train_reconstruction_srae] = srae.predict(train_images, batch_size=train_images.shape[0])
    
########################################################################################################################
if visualize_embedding:
    # Set list of possible attribute values and index of attribute to color-code for visualization
    if dataset is 'rotation-scaling':
        attr_values = list(np.arange(10, 10+n_scales*int(scale_step*10), int(scale_step*10))/10)
        visualize_attr = 1
        
    elif dataset is 'rotation-identity':
        attr_values = list(range(n_ids))
        visualize_attr = 0
        
    elif dataset is 'position-rotation':
        attr_values = list(np.arange(0, 360, angle_step))
        visualize_attr = 0
    
    elif dataset is 'position-identity':
        attr_values = list(range(n_ids))
        visualize_attr = 0

    # Visualize embedding in a 3D-scatter plot
    plotter = Plotter() # Instantiate plotter
    plotter.plot_embedding(train_items, train_embedding_srae, attr_values, visualize_attr)

########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file contains the code for the separated extraction of latent factors in Section 3.4.3 of the paper.
# It provides the What-Where Encoder-Decoder model and the generation of moving sequence datasets with
# variation in position and identity.
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
    print("[" + datetime.now().strftime("%H:%M:%S") + "][Separated extraction] " + message)

import time
execution_time = time.time()

import sys
from keras.datasets import mnist
import numpy as np

from core.utils import *
from core.sequence_generator import Item, SequenceGenerator
from models import What_Where_Encoder_Decoder
from core.powersfa import ordered_gsfa_loss, GeneralizedBatchGenerator
from core.plotter import Plotter

# Fixing seeds for comparison
numpy_seed = 0
np.random.seed(numpy_seed)

# Verification of the directory structure required for the storage of the results
check_dir_structure()

######################################### Setup ########################################################################
# Model Configuration
n_what_features = 1                     # Dimensionality of the latent space of the What-Encoder
n_where_features = 2                    # Dimensionality of the latent space of the Where-Encoder
encoder_loss = ordered_gsfa_loss        # Loss function used to train the encoder
decoder_loss = 'binary_crossentropy'    # Loss function used to train the decoder
optimizer = 'nadam'                     # Optimizer used for training

# Training Parameters
batch_size = 1/5                        # Fraction of images that form a batch
n_epochs = 1000                         # Number of epochs to train the model
train_what_encoder = False              # Train the What-Encoder
train_where_encoder = False             # Train the Where-Encoder
train_decoder = False                   # Train the Decoder
save_weights = False                    # Save model weights

plot_emb_recon = True                   # Visualize the computed embedding in a 3D-plot and plot reconstructions

######################################### Dataset generation ###########################################################
log("Generating position-identity dataset.")
# Loading MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Dataset configuration
moving_sequence = True      # Flag indicating whether the resulting sequence is a moving sequence
n_ids = 10                  # Number of different identities
n_positions = 36            # Number of positions per dimension
pos_step = 1                # Stepwidth between each position
shuffle = True              # Shuffle the items

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

if train_what_encoder:
    for i in range(n_items):
        for j in range(n_items):
            if train_items[i].attributes[0] == train_items[j].attributes[0]:
                W[i,j] = W[j,i] = 1

elif train_where_encoder:
    for i in range(n_items):
        for j in range(n_items):
            if (abs(train_items[i].attributes[1][0] - train_items[j].attributes[1][0]) <= 2*pos_step and abs(train_items[i].attributes[1][1] - train_items[j].attributes[1][1]) <= 2*pos_step):
                W[i,j] = W[j,i] = 1

train_images = val_images = np.array([item.image for item in train_items])
train_similarity = val_similarity = W

######################################### What-Where Encoder-Decoder model #############################################
log("Configuring the What-Where Encoder-Decoder model.")
if train_what_encoder:
    return_enc = 0
else:
    return_enc = 1
ed_model = What_Where_Encoder_Decoder(n_what_features, n_where_features, return_enc)
encoder, decoder = ed_model.encoder, ed_model.decoder
encoder.compile(loss=encoder_loss, optimizer=optimizer)
decoder.compile(loss=decoder_loss, optimizer=optimizer)

print(encoder.summary())
print(decoder.summary())

# Train the Encoder model
if train_what_encoder or train_where_encoder:
    # Prepare a batch generator for the training and validation data
    train_gen = GeneralizedBatchGenerator(train_images, train_similarity/int(train_images.shape[0]*batch_size), int(train_images.shape[0]*batch_size))
    val_gen = GeneralizedBatchGenerator(val_images, val_similarity/int(train_images.shape[0]*batch_size), int(train_images.shape[0]*batch_size))

    if train_what_encoder:
        log("Start training the What-Encoder model.")
        history_encoder = encoder.fit_generator(train_gen, int(1/batch_size), epochs=n_epochs, validation_data=val_gen, validation_steps=len(val_gen))
        if save_weights:
            log("Saving trained What-Encoder model weights.")
            encoder.save_weights('Run/what_encoder_weights.h5')
        log("Computing and storing What-Encoder predictions.")
        train_what_embedding = encoder.predict(train_images, batch_size=train_images.shape[0])
        np.save('Run/train_what_embedding.npy', train_what_embedding)
        
    else:
        log("Start training the Where-Encoder model.")
        history_encoder = encoder.fit_generator(train_gen, int(1/batch_size), epochs=n_epochs, validation_data=val_gen, validation_steps=len(val_gen))
        if save_weights:
            log("Saving trained Where-Encoder model weights.")
            encoder.save_weights('Run/where_encoder_weights.h5')
        log("Computing and storing Where-Encoder predictions.")
        train_where_embedding = where_encoder.predict(train_images, batch_size=train_images.shape[0])
        np.save('Run/train_where_embedding.npy', train_where_embedding)
    
    sys.exit("The encoder has been trained successfully. To train or use the decoder, please set train_what_encoder and train_where_encoder to False.")
    
else:   # Alternatively load What- and Where-embeddings
    log("Loading stored What- and Where-embeddings.")
    train_what_embedding = np.load('pretrained_models/separated_extraction/train_what_embedding.npy', allow_pickle=True)
    train_where_embedding = np.load('pretrained_models/separated_extraction/train_where_embedding.npy', allow_pickle=True)
    
# Combine What and Where embeddings into single latent samples
train_embedding = val_embedding = np.array([[train_where_embedding[idx, 0], train_where_embedding[idx, 1], train_what_embedding[idx, 0]] for idx in range(train_where_embedding.shape[0])])

# Train the Decoder model
if train_decoder:
    log("Start training the Decoder model.")
    history_decoder = decoder.fit(x=train_embedding, y=train_images, batch_size=int(train_images.shape[0]*batch_size), epochs=n_epochs, validation_data=(val_embedding, val_images))
    if save_weights:
        log("Saving trained Decoder model weights.")
        decoder.save_weights('Run/decoder_weights.h5')

else:   # Alternatively load Decoder model weights
    log("Loading stored Decoder model weights.")
    decoder.load_weights('pretrained_models/separated_extraction/decoder_weights.h5')
    
log("Computing decoder predictions.")
train_reconstruction = decoder.predict(train_embedding, batch_size=train_embedding.shape[0])
    
########################################################################################################################
if plot_emb_recon:
    plotter = Plotter() # Instantiate plotter

    # Visualize embedding in a 3D-scatter plot
    plotter.plot_embedding(train_items, train_embedding, list(range(n_ids)), 0)
    
    # Plot input and reconstructed images
    plotter.plot_comp_input_reconstruction(train_images, train_reconstruction)

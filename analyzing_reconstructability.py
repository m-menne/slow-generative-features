########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file contains the code for the experiments in Section 3.3 of the paper.
# It provides the Encoder-Decoder model, Slowness-Regularized Autoencoder model with different loss weightings
# as well as the Autoencoder model with whitened latent factors and the generation of moving sequence datasets with
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
    print("[" + datetime.now().strftime("%H:%M:%S") + "][Analyzing reconstructability] " + message)
    
import time
execution_time = time.time()

from keras.datasets import mnist

import numpy as np

from core.utils import *
from core.moving_sequence_generator import MovingSequenceGenerator
from models import Encoder_Decoder, SRAE
from core.powersfa import ordered_gsfa_loss, GeneralizedBatchGenerator, GeneralizedBatchGenerator2
from core.plotter import Plotter

# Fixing seeds for comparison
numpy_seed = 0
np.random.seed(numpy_seed)

# Verification of the directory structure required for the storage of the results
check_dir_structure()

######################################### Setup ########################################################################
# Sequence Generation
n_seq = 8001                            # Total number of sequences
seq_length = 5                          # Number of images per sequence
velocity = 0.05                         # Velocity of the moving images
start_position = None                   # Starting position of the sequences
connected_seq = True                    # Start position of a sequence is equal to the end position of the last sequence

# Dataset Processing
data_split = [0.6, 0.2]                 # Ratio of training and validation data. The rest is used as test data
normalization = False                   # Normalizing training images to zero mean and std. dev. of 1
rescaling = True                        # Scaling image data to [0., 1.] interval
shuffle = True                          # Shuffle the generated data

# Model Configuration
model = 'SRAE'                          # Set model: Encoder-Decoder, SRAE or WAE
n_features = 5                          # Dimensionality of the latent space
encoder_loss = ordered_gsfa_loss        # Loss function used to train the encoder
decoder_loss = 'binary_crossentropy'    # Loss function used to train the decoder
optimizer = 'nadam'                     # Optimizer used for training

# Training Parameters
batch_size = 1/5                        # Fraction of images that form a batch
n_epochs = 1000                         # Number of epochs to train the model
train_model = False                     # Train the model (True) or load model weights (False)
save_weights = False                    # Save model weights
save_data = True                        # Save data which is essential for plotting

plot_recon_comp = False                 # Plot reconstruction losses and reconstructions in comparison

######################################### Dataset generation ###########################################################
log("Generating the moving sequence dataset.")
# Loading MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select input images
input_images = x_train[:3]

# Initialize the MovingSequenceGenerator and generate the datasets
seq_gen = MovingSequenceGenerator(input_images, n_seq, seq_length, velocity=velocity, start_position=start_position, connected_seq=connected_seq)
train_images, train_positions, train_indices, train_similarity, val_images, val_positions, val_indices, val_similarity, test_images, test_positions, test_indices, test_similarity = seq_gen.generateDataset(data_split, normalization, rescaling, shuffle)

######################################### Encoder-Decoder model ########################################################
if model is 'Encoder-Decoder':
    log("Configuring the Encoder-Decoder model.")
    ed_model = Encoder_Decoder(n_features)
    encoder, decoder = ed_model.encoder, ed_model.decoder
    encoder.compile(loss=encoder_loss, optimizer=optimizer)
    decoder.compile(loss=decoder_loss, optimizer=optimizer)

    print(encoder.summary())
    print(decoder.summary())

    # Train the Encoder model
    if train_model:
        # Prepare a batch generator for the training and validation data
        train_gen = GeneralizedBatchGenerator(train_images, train_similarity/int(train_images.shape[0]*batch_size), int(train_images.shape[0]*batch_size))
        val_gen = GeneralizedBatchGenerator(val_images, val_similarity/int(train_images.shape[0]*batch_size), int(train_images.shape[0]*batch_size))

        log("Start training the Encoder model.")
        history_encoder = encoder.fit_generator(train_gen, int(1/batch_size), epochs=n_epochs, validation_data=val_gen, validation_steps=len(val_gen))
        if save_weights:
            log("Saving trained Encoder model weights.")
            encoder.save_weights('Run/encoder_weights.h5')

    else:   # Alternatively load encoder model weights
        log("Loading stored Encoder model weights.")
        encoder.load_weights('pretrained_models/analyzing_reconstructability/encoder_weights.h5')

    log("Computing encoder predictions.")
    train_embedding_ed = encoder.predict(train_images, batch_size=train_images.shape[0])
    val_embedding_ed = encoder.predict(val_images, batch_size=val_images.shape[0])
    test_embedding_ed = encoder.predict(test_images, batch_size=test_images.shape[0])

    # Train the Decoder model
    if train_model:
        log("Start training the Decoder model.")
        history_decoder = decoder.fit(x=train_embedding_ed, y=train_images, batch_size=int(train_images.shape[0]*batch_size), epochs=n_epochs, validation_data=(val_embedding_ed, val_images))
        if save_weights:
            log("Saving trained Decoder model weights.")
            decoder.save_weights('Run/decoder_weights.h5')

    else:   # Alternatively load Decoder model weights
        log("Loading stored Decoder model weights.")
        decoder.load_weights('pretrained_models/analyzing_reconstructability/decoder_weights.h5')
        
    log("Computing decoder predictions.")
    train_reconstruction_ed = decoder.predict(train_embedding_ed, batch_size=train_embedding_ed.shape[0])
    test_reconstruction_ed = decoder.predict(test_embedding_ed, batch_size=test_embedding_ed.shape[0])
    
    if train_model and save_data:
        log("Save reconstruction losses and reconstructions to file.")
        np.save('Run/recon_data_ed.npy', [history_decoder.history['loss'], train_reconstruction_ed, test_reconstruction_ed])

######################################### Slowness-Regularized Autoencoder model #######################################
if model is 'SRAE':
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
        srae.load_weights('pretrained_models/analyzing_reconstructability/srae_weights.h5')

    log("Computing predictions.")
    [train_embedding_srae, train_reconstruction_srae] = srae.predict(train_images, batch_size=train_images.shape[0])
    [test_embedding_srae, test_reconstruction_srae] = srae.predict(test_images, batch_size=test_images.shape[0])
    
    if train_model and save_data:
        log("Save reconstruction losses and reconstructions to file.")
        np.save('Run/recon_data_srae.npy', [history_srae.history['reconstruction_loss'], train_reconstruction_srae, test_reconstruction_srae])

######################################### Autoencoder model (whitened latent factors) ##################################
if model is 'WAE':
    log("Configuring the WAE model.")
    wae = SRAE(n_features).model
    wae.compile(loss=[encoder_loss, decoder_loss], loss_weights=[0., 1.], optimizer=optimizer)

    print(wae.summary())

    # Train the WAE model
    if train_model:
        # Prepare a batch generator for the training and validation data
        train_gen = GeneralizedBatchGenerator2(train_images, train_similarity/int(train_images.shape[0]*batch_size), int(train_images.shape[0]*batch_size))
        val_gen = GeneralizedBatchGenerator2(val_images, val_similarity/int(train_images.shape[0]*batch_size), int(train_images.shape[0]*batch_size))
        
        log("Start training the WAE model.")
        history_wae = wae.fit_generator(train_gen, int(1/batch_size), epochs=n_epochs, validation_data=val_gen, validation_steps=len(val_gen))
        if save_weights:
            log("Saving trained WAE model weights.")
            wae.save_weights('Run/wae_weights.h5')

    else:   # Alternatively load WAE model weights
        log("Loading stored WAE model weights.")
        wae.load_weights('pretrained_models/analyzing_reconstructability/wae_weights.h5')

    log("Computing predictions.")
    [train_embedding_wae, train_reconstruction_wae] = wae.predict(train_images, batch_size=train_images.shape[0])
    [test_embedding_wae, test_reconstruction_wae] = wae.predict(test_images, batch_size=test_images.shape[0])
    
    if train_model and save_data:
        log("Save reconstruction losses and reconstructions to file.")
        np.save('Run/recon_data_wae.npy', [history_wae.history['reconstruction_loss'], train_reconstruction_wae, test_reconstruction_wae])

########################################################################################################################
if plot_recon_comp:
    log("Load reconstruction losses and reconstructions of the Encoder-Decoder, SRAE and WAE model")
    recon_data_ed = np.load('Run/recon_data_ed.npy', allow_pickle=True)
    recon_data_srae = np.load('Run/recon_data_srae.npy', allow_pickle=True)
    recon_data_wae = np.load('Run/recon_data_wae.npy', allow_pickle=True)
        
    plotter = Plotter() # Instantiate plotter
        
    # Plot reconstruction losses of the Encoder-Decoder, SRAE and WAE model in comparison
    plotter.plot_recon_learning_curve_comp_ed_srae_wae(recon_data_ed[0], recon_data_srae[0], recon_data_wae[0])

    # Plot the reconstructions of the Encoder-Decoder, SRAE and WAE model in comparison
    plotter.plot_recon_comp_ed_srae_wae(train_images, recon_data_ed[1], recon_data_srae[1], recon_data_wae[1])

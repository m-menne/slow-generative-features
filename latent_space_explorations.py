########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file contains the code for the latent space explorations in Section 3.4.1 of the paper.
# It provides the Encoder-Decoder model, Slowness-Regularized Autoencoder model with different loss weightings
# as well as the Autoencoder model with and without whitened latent factors. The used dataset consists of
# static sequences with only a variation in identity.
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
    print("[" + datetime.now().strftime("%H:%M:%S") + "][Latent space explorations] " + message)
    
import time
execution_time = time.time()

from keras.datasets import mnist

import numpy as np

from core.utils import *
from core.moving_sequence_generator import MovingSequenceGenerator
from models import Encoder_Decoder, SRAE, AE
from core.powersfa import ordered_gsfa_loss, GeneralizedBatchGenerator, GeneralizedBatchGenerator2, GeneralizedBatchGenerator3
from core.plotter import Plotter

# Fixing seeds for comparison
numpy_seed = 0
np.random.seed(numpy_seed)

# Verification of the directory structure required for the storage of the results
check_dir_structure()

######################################### Setup ########################################################################
# Sequence Generation
n_seq = 10                              # Total number of sequences
seq_length = 5                          # Number of images per sequence
velocity = 0.0                          # Velocity of the moving images
start_position = [0.5, 0.5]             # Starting position of the sequences
connected_seq = True                    # Start position of a sequence is equal to the end position of the last sequence

# Dataset Processing
data_split = [1.0, 0.0]                 # Ratio of training and validation data. The rest is used as test data
normalization = False                   # Normalizing training images to zero mean and std. dev. of 1
rescaling = True                        # Scaling image data to [0., 1.] interval
shuffle = False                         # Shuffle the generated data

# Model Configuration
model = 'AE'               # Set model: Encoder-Decoder, SRAE, WAE or AE
n_features = 2                          # Dimensionality of the latent space
encoder_loss = ordered_gsfa_loss        # Loss function used to train the encoder
decoder_loss = 'binary_crossentropy'    # Loss function used to train the decoder
optimizer = 'nadam'                     # Optimizer used for training

# Training Parameters
batch_size = 1/5                        # Fraction of images that form a batch
n_epochs = 200                          # Number of epochs to train the model
train_model = False                     # Train the model (True) or load model weights (False)
save_weights = False                    # Save model weights

latent_space_exploration = True         # Perfom and plot a latent space exploration on the trained models

######################################### Dataset generation ###########################################################
log("Generating the dataset with variation in identity.")
# Loading MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select input images
used_ids = np.arange(10)
images = [[x_train[i] for i in range(x_train.shape[0]) if y_train[i] == used_ids[j]] for j in range(10)]

# Compute the cross-entropy of all images with a certain identity to the first one with this identity
similarities = [[(cross_entropy(np.reshape(images[i][0], (28*28,)), np.reshape(images[i][j], (28*28,))), j) for j in range(len(images[i]))] for i in range(len(images))]
# Sort ascending w.r.t. the cross-entropy
for i in range(10):
    similarities[i].sort(key=lambda tup: abs(tup[0]))
# Take the most similiar images w.r.t. to the first one in each class
idx = [[similarities[i][j][1] for j in range(seq_length)] for i in range(10)]
input_images = [[images[i][j] for j in idx[i]] for i in range(10)]

# Initialize the MovingSequenceGenerator and generate the datasets
seq_gen = MovingSequenceGenerator(input_images, n_seq, seq_length, velocity=velocity, start_position=start_position, connected_seq=connected_seq, variation_mode=True)
train_images, train_positions, train_indices, train_similarity, _, _, _, _, _, _, _, _ = seq_gen.generateDataset(data_split, normalization, rescaling, shuffle)

val_images = test_images = train_images
val_similarity = test_similarity = train_similarity

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
        encoder.load_weights('pretrained_models/latent_space_explorations/encoder_weights.h5')

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
        decoder.load_weights('pretrained_models/latent_space_explorations/decoder_weights.h5')
        
    log("Computing decoder predictions.")
    train_reconstruction_ed = decoder.predict(train_embedding_ed, batch_size=train_embedding_ed.shape[0])
    test_reconstruction_ed = decoder.predict(test_embedding_ed, batch_size=test_embedding_ed.shape[0])
    
######################################### Slowness-Regularized Autoencoder model #######################################
if model is 'SRAE':
    log("Configuring the SRAE model.")
    srae_model = SRAE(n_features)
    srae = srae_model.model
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
        srae.load_weights('pretrained_models/latent_space_explorations/srae_weights.h5')

    log("Computing predictions.")
    [train_embedding_srae, train_reconstruction_srae] = srae.predict(train_images, batch_size=train_images.shape[0])
    [test_embedding_srae, test_reconstruction_srae] = srae.predict(test_images, batch_size=test_images.shape[0])
    
    if latent_space_exploration:
        decoder = srae_model.decoder
        decoder.compile(loss=decoder_loss, optimizer=optimizer)
    
######################################### Autoencoder model (whitened latent factors) ##################################
if model is 'WAE':
    log("Configuring the WAE model.")
    wae_model = SRAE(n_features)
    wae = wae_model.model
    wae.compile(loss=[encoder_loss, decoder_loss], loss_weights=[0., 1.], optimizer=optimizer)

    print(wae.summary())

    # Train the AE model
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
        wae.load_weights('pretrained_models/latent_space_explorations/wae_weights.h5')

    log("Computing predictions.")
    [train_embedding_wae, train_reconstruction_wae] = wae.predict(train_images, batch_size=train_images.shape[0])
    [test_embedding_wae, test_reconstruction_wae] = wae.predict(test_images, batch_size=test_images.shape[0])
    
    if latent_space_exploration:
        decoder = wae_model.decoder
        decoder.compile(loss=decoder_loss, optimizer=optimizer)
        
######################################### Autoencoder model ############################################################
if model is 'AE':
    log("Configuring the AE model.")
    ae_model = AE(n_features)
    ae = ae_model.model
    ae.compile(loss=decoder_loss, optimizer=optimizer)

    print(ae.summary())
    
    # Train the AE model
    if train_model:
        # Prepare a batch generator for the training and validation data
        train_gen = GeneralizedBatchGenerator3(train_images, int(train_images.shape[0]*batch_size))
        val_gen = GeneralizedBatchGenerator3(val_images, int(train_images.shape[0]*batch_size))

        log("Start training the AE model.")
        history_ae = ae.fit_generator(train_gen, int(1/batch_size), epochs=n_epochs, validation_data=val_gen, validation_steps=len(val_gen))
        if save_weights:
            log("Saving trained AE model weights.")
            ae.save_weights('Run/ae_weights.h5')

    else:   # Alternatively load AE model weights
        log("Loading stored AE model weights.")
        ae.load_weights('pretrained_models/latent_space_explorations/ae_weights.h5')

    log("Computing predictions.")
    train_embedding_ae = ae_model.encoder.predict(train_images, batch_size=train_images.shape[0])
    test_embedding_ae = ae_model.encoder.predict(test_images, batch_size=test_images.shape[0])

    train_reconstruction_ae = ae.predict(train_images, batch_size=train_images.shape[0])
    test_reconstruction_ae = ae.predict(test_images, batch_size=test_images.shape[0])
    
    if latent_space_exploration:
        decoder = ae_model.decoder
        decoder.compile(loss=decoder_loss, optimizer=optimizer)
    
########################################################################################################################
if latent_space_exploration:
    log("Computing and assigning feature values for latent space exploration.")
    # Compute feature values in a grid structure
    min = [-2., -2.]    # Minima of each dimension
    max = [2., 2.]      # Maxima of each dimension
    scale = 50.         # Set scaling of the grid
    size = np.array((np.abs(max)+np.abs(min))*scale).astype(int) # Compute number of feature values
    threshold = 5.      # Set threshold for correspondences

    feature_values = np.array([[min[0] + float(i) * (1./scale), min[1] + float(j) * (1/scale), -1., -1.] for i in range(size[0]) for j in range(size[1])])

    # Decode feature values
    predicted_images = decoder.predict(feature_values[:,:2], batch_size=feature_values.shape[0])

    def get_label(org_images, predicted_img, threshold):
        """Compute best fitting label to predicted image corresponding to the cross entropy error.
        """
        errors = [cross_entropy(org_img, predicted_img) for org_img in org_images]
        min = np.min(errors)
        if min < threshold:
            return np.argmin(errors)//5., min
        else:
            return -1, -1.

    # Assign labels to feature values
    for idx, img in enumerate(predicted_images):
        feature_values[idx,-2:] = get_label(train_images, img, threshold)
        
    # Rescale cross entropy values to range [0, 1] for plotting
    feature_values[:,-1] += 1.
    feature_values[:,-1] /= np.max(feature_values[:,-1])

    log("Plotting latent space exploration.")
    plotter = Plotter() # Instantiate plotter
    plotter.plot_latent_space_exploration(feature_values)

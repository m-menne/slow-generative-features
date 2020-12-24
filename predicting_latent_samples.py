########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file contains the code for the prediction of latent samples in Section 3.5.2 of the paper.
# It provides the Encoder-Predictor-Decoder model and the generation of moving sequence datasets with
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
    print("[" + datetime.now().strftime("%H:%M:%S") + "][Predicting latent samples] " + message)
    
import time
execution_time = time.time()

from keras.datasets import mnist

import numpy as np

from core.utils import *
from core.moving_sequence_generator import MovingSequenceGenerator
from models import Encoder_Predictor_Decoder
from core.powersfa import ordered_gsfa_loss, GeneralizedBatchGenerator
from core.plotter import Plotter

# Fixing seeds for comparison
numpy_seed = 0
np.random.seed(numpy_seed)

# Verification of the directory structure required for the storage of the results
check_dir_structure()

######################################### Setup ########################################################################
# Sequence Generation
n_seq = 9000                            # Total number of sequences
seq_length = 20                         # Number of images per sequence
velocity = 0.05                         # Velocity of the moving images
start_position = None                   # Starting position of the sequences
connected_seq = True                    # Start position of a sequence is equal to the end position of the last sequence

# Dataset Processing
data_split = [0.6, 0.2]                 # Ratio of training and validation data. The rest is used as test data
normalization = False                   # Normalizing training images to zero mean and std. dev. of 1
rescaling = True                        # Scaling image data to [0., 1.] interval
shuffle = False                         # Shuffle the generated data

# Model Configuration
n_features = 3                          # Dimensionality of the latent space
encoder_loss = ordered_gsfa_loss        # Loss function used to train the encoder
predictor_loss = 'mae'                  # Loss function used to train the predictor
decoder_loss = 'binary_crossentropy'    # Loss function used to train the decoder
optimizer = 'nadam'                     # Optimizer used for training the encoder and decoder
predictor_optimizer = 'rmsprop'         # Optimizer used for training the predictor

# Training Parameters
batch_size = 1/5                        # Fraction of images that form a batch
n_epochs = 1000                         # Number of epochs to train the model
train_encoder_model = False             # Train the encoder model
load_encoder_model = False              # Load the encoder model
train_prediction_model = False          # The the prediction model
train_decoder_model = False             # Train the decoder model
save_weights = False                    # Save model weights

plot_input_prediction = True            # Plot input, ground truth and predicted images in comparison

######################################### Dataset generation ###########################################################
log("Generating the moving sequence dataset.")
# Loading MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select input images
used_ids = np.arange(10)
# Get first image in the MNIST dataset of every used identity
images = [[x_train[i] for i in range(x_train.shape[0]) if y_train[i] == used_ids[j]] for j in range(used_ids.shape[0])]
input_images = np.array([img[0] for img in images])

# Initialize the MovingSequenceGenerator and generate the datasets
seq_gen = MovingSequenceGenerator(input_images, n_seq, seq_length, velocity=velocity, start_position=start_position, connected_seq=connected_seq)
train_images, train_positions, train_indices, train_similarity, val_images, val_positions, val_indices, val_similarity, test_images, test_positions, test_indices, test_similarity = seq_gen.generateDataset(data_split, normalization, rescaling, shuffle)
 
######################################### Encoder-Predictor-Decoder model ##############################################

log("Configuring the Encoder-Predictor-Decoder model.")
epd_model = Encoder_Predictor_Decoder(n_features, seq_length//2, seq_length//2)
encoder, predictor, decoder = epd_model.encoder, epd_model.predictor, epd_model.decoder
encoder.compile(loss=encoder_loss, optimizer=optimizer)
predictor.compile(loss=predictor_loss, optimizer=predictor_optimizer)
decoder.compile(loss=decoder_loss, optimizer=optimizer)

print(encoder.summary())
print(predictor.summary())
print(decoder.summary())


# Train the Encoder model
if train_encoder_model:
    # Prepare a batch generator for the training and validation data
    train_gen = GeneralizedBatchGenerator(train_images, train_similarity/int(train_images.shape[0]*batch_size), int(train_images.shape[0]*batch_size))
    val_gen = GeneralizedBatchGenerator(val_images, val_similarity/int(train_images.shape[0]*batch_size), int(train_images.shape[0]*batch_size))

    log("Start training the Encoder model.")
    history_encoder = encoder.fit_generator(train_gen, int(1/batch_size), epochs=n_epochs, validation_data=val_gen, validation_steps=len(val_gen))
    if save_weights:
        log("Saving trained Encoder model weights.")
        encoder.save_weights('Run/encoder_weights.h5')

elif load_encoder_model:   # Alternatively load encoder model weights
    log("Loading stored Encoder model weights.")
    encoder.load_weights('Run/encoder_weights.h5')
    
else:
    log("Loading stored embeddings.")
    train_embedding = np.load('pretrained_models/predicting_latent_samples/train_embedding.npy', allow_pickle=True)
    val_embedding = np.load('pretrained_models/predicting_latent_samples/val_embedding.npy', allow_pickle=True)
    test_embedding = np.load('pretrained_models/predicting_latent_samples/test_embedding.npy', allow_pickle=True)
    
if train_encoder_model or load_encoder_model:
    log("Computing encoder predictions.")
    train_embedding = encoder.predict(train_images, batch_size=train_images.shape[0])
    val_embedding = encoder.predict(val_images, batch_size=val_images.shape[0])
    test_embedding = encoder.predict(test_images, batch_size=test_images.shape[0])
   

# Train the Prediction model
n_history = n_predictions = seq_length//2
# Generate prediction datasets
train_history, train_prediction = generate_prediction_dataset(train_embedding, seq_length, n_history, n_predictions)
val_history, val_prediction = generate_prediction_dataset(val_embedding, seq_length, n_history, n_predictions)
test_history, test_prediction = generate_prediction_dataset(test_embedding, seq_length, n_history, n_predictions)

if train_prediction_model:
    log("Start training the Prediction model.")
    history_predictor = predictor.fit(x=train_history, y=train_prediction, batch_size=batch_size, epochs=n_epochs, validation_data=(val_history, val_prediction))
    if save_weights:
        log("Saving trained Prediction model weights.")
        predictor.save_weights('Run/predictor_weights.h5')

else:   # Alternatively load Prediction model weights
    log("Loading stored Prediction model weights.")
    predictor.load_weights('pretrained_models/predicting_latent_samples/predictor_weights.h5')

log("Computing forecasts with the prediction model.")
train_forecast = predictor.predict(train_history)
train_predicted_embedding = merge_history_forecast(train_history, train_forecast)
train_predicted_embedding = np.reshape(train_predicted_embedding, (train_images.shape[0], n_features))

val_forecast = predictor.predict(val_history)
val_predicted_embedding = merge_history_forecast(val_history, val_forecast)
val_predicted_embedding = np.reshape(val_predicted_embedding, (val_images.shape[0], n_features))

test_forecast = predictor.predict(test_history)
test_predicted_embedding = merge_history_forecast(test_history, test_forecast)
test_predicted_embedding = np.reshape(test_predicted_embedding, (test_images.shape[0], n_features))


# Train the Decoder model
if train_decoder_model:
    log("Start training the Decoder model.")
    history_decoder = decoder.fit(x=train_predicted_embedding, y=train_images, batch_size=int(train_images.shape[0]*batch_size), epochs=n_epochs, validation_data=(val_predicted_embedding, val_images))
    if save_weights:
        log("Saving trained Decoder model weights.")
        decoder.save_weights('Run/decoder_weights.h5')

else:   # Alternatively load Decoder model weights
    log("Loading stored Decoder model weights.")
    decoder.load_weights('pretrained_models/predicting_latent_samples/decoder_weights.h5')
    
log("Computing decoder predictions.")
train_reconstruction = decoder.predict(train_embedding, batch_size=train_embedding.shape[0])
test_reconstruction = decoder.predict(test_embedding, batch_size=test_embedding.shape[0])
train_predicted_reconstruction = decoder.predict(train_predicted_embedding)
test_predicted_reconstruction = decoder.predict(test_predicted_embedding)

########################################################################################################################
if plot_input_prediction:
    # Plot input, ground truth and predicted images in comparison
    plotter = Plotter() # Instantiate plotter
    plotter.plot_comp_input_ground_prediction(train_images[:20], train_predicted_reconstruction[:20])

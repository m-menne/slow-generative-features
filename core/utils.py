########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file contains several other useful functions.
########################################################################################################################

import numpy as np

def check_dir_structure():
    """Check whether the directory structure required for program execution exists. If this is not the case, one is created.
    """
    import os
    if not os.path.isdir('Run'):
        os.mkdir('Run')

def cross_entropy(input_a, input_b):
    """Computes the cross entropy between the two input images.
    """
    return -np.sum(input_b*np.log(input_a+1e-9))/input_a.shape[0]

def generate_prediction_dataset(data, seq_length, n_history, n_predictions):
    """Generates a dataset which can be used for training the prediction model.
    Args:
        data:           Data from which the dataset should be build
        n_history:      Number of data samples that should be used for input
        n_predictions:  Number of data samples that the model should predict
    Return:
        x, y:   Input and output data for the prediction model
    """
    x = [data[(i*seq_length):(i*seq_length) + n_history] for i in range(len(data)//seq_length)]
    y = [data[(i*seq_length) + n_history:(i*seq_length) + n_history + n_predictions] for i in range(len(data)//seq_length)]
    
    return np.array(x), np.array(y)
    
def merge_history_forecast(history, forecast):
    """Merges history and forecast data into one sequence.
    """
    return np.array([list(history[i]) + list(forecast[i]) for i in range(len(history))])

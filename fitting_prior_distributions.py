########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file contains the code for fitting the defined prior distributions in Section 3.5.1 of the paper.
# It provides the fitting procedures for the rectangular and conical frustum for embedded datasets with variation in
# position and identity as well as rotation and scaling.
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
    print("[" + datetime.now().strftime("%H:%M:%S") + "][Fitting prior distributions] " + message)

import time
execution_time = time.time()

import numpy as np

from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from scipy.stats import *

from core.utils import *
from models import Encoder_Decoder
from core.plotter import Plotter

# Fixing seeds for comparison
numpy_seed = 0
np.random.seed(numpy_seed)

# Verification of the directory structure required for the storage of the results
check_dir_structure()

######################################### Setup ########################################################################
fitting_mode = "conical_frustum"    # Select fitting mode: rectangular_frustum or conical frustum
n_features = 3                          # Dimensionality of the latent space
n_samples = 5000                        # Number of samples to draw from the fitted frustum

plot_emb_recon = True                   # Visualize embedded and fitted in a scatter 3D-plot and plot reconstructions

########################################################################################################################
# Load embedded data for fitting the distribution
if fitting_mode is "rectangular_frustum":
    emb_data = np.load('pretrained_models/fitting_prior_distributions/embedding_rectangular.npy')
else:
    emb_data = np.load('pretrained_models/fitting_prior_distributions/embedding_conical.npy')

# Rotate embedding with FastICA to align it to the coordinate axes
ica = FastICA(n_features, whiten=True)
emb_data = ica.fit_transform(emb_data)

# Identify categorical dimension by computing the variance of absolute occurrences
hists = [np.histogram(emb_data[:, i], bins=100)[0] for i in range(n_features)]
print("Dimension of the categorical data: " + str(np.argmax(np.var(hists, axis=1))))

log("Starting fitting mode: " + str(mode))

######################################### Fitting the rectangular frustum ##############################################
if fitting_mode == "rectangular_frustum":
    
    ### Estimate the parameters of the rectangular frustum from the embedding ###
    
    # Identify categorical dimensions by computing the variance of the frequencies of the discretized latent points
    hists = [np.histogram(emb_data[:, i], bins=100)[0] for i in range(n_features)]
    orientation = np.argmax(np.var(hists, axis=1))
   
    # Find the discrete values along the categorical dimension by using the k-Means clustering algorithm
    n_cat = 10
    kmeans = KMeans(n_clusters=n_cat, random_state=0).fit(emb_data[:, orientation].reshape(-1, 1))
    cat_values = kmeans.cluster_centers_
   
    # Set the range for the categorical dimension
    cat_dim_range = [np.amin(cat_values), np.amax(cat_values)]
    
    # Set the number of latent points used for parameter estimation of the base sides
    amount_ref = 0.1
    n_ref_points = int(emb_data.shape[0] * amount_ref)
    
    # Select latent points that are on the edge of the embedding with respect to all dimensions
    indices = np.array([[np.argpartition(emb_data[:, dim], n_ref_points)[:n_ref_points], np.argpartition(-emb_data[:, dim], n_ref_points)[:n_ref_points]]  for dim in range(n_features)])
    selected_lat_points = np.array([[[emb_data[idx] for idx in indices[dim, i]]  for i in range(2)] for dim in range(n_features)])
    
    # Compute the minima and maxima of the selected latent points of the embedded data for each dimension
    extrems =  np.array([[[np.amin(selected_lat_points[dim, i], axis=0), np.amax(selected_lat_points[dim, i], axis=0) ] for i in range(2)] for dim in range(n_features)])
    
    # Set parameters for the base sides
    if orientation == 0:
        base_points = [extrems[orientation, i, :, [1, 2]] for i in range(2)]
    elif orientation == 1:
        base_points = [extrems[orientation, i, :, [0, 2]] for i in range(2)]
    else:
        base_points = [extrems[orientation, i, :, [0, 1]] for i in range(2)]
    base_points = np.reshape(base_points, (2,2,2))

    # Swap array entries to order minima and maxima
    for i in range(2):
        tmp = base_points[i, 0, 1]
        base_points[i, 0, 1] = base_points[i, 1, 0]
        base_points[i, 1, 0] = tmp
    
    
    ### Draw samples using the estimated parameters ###
    
    # Draw uniform samples along the categorical dimension
    discrete_samples = np.random.randint(0, n_cat, n_samples)
    cat_dim_samples = [cat_values[sample][0] for sample in discrete_samples]
    cat_dim_percentage = [(sample-cat_dim_range[0])/(cat_dim_range[1]-cat_dim_range[0])for sample in cat_dim_samples]
    
    # Compute the dimensions of the rectangular layers for the drawn categorical samples by interpolating between the base sides
    slope = base_points[1] - base_points[0]
    interpolated_ranges = np.array([cat_dim_percentage[idx] *  slope + base_points[0]  for idx, sample in enumerate(cat_dim_samples)])
    interpolated_ranges = np.reshape(interpolated_ranges, (n_samples, 2, 2))
    
    # Draw uniform samples for the rectangular layers
    dep_dim_samples = np.array([np.random.uniform(low=limits[0], high=limits[1], size=(2,)) for limits in interpolated_ranges])
    
    # Combine the drawn samples to latent points
    if orientation == 0:
        fitted_data = np.array([[cat_dim_samples[idx], dep_dim_samples[idx, 0], dep_dim_samples[idx, 1]] for idx in range(n_samples)])
    elif orientation == 1:
        fitted_data = np.array([[dep_dim_samples[idx, 0], cat_dim_samples[idx], dep_dim_samples[idx, 1]] for idx in range(n_samples)])
    else:
        fitted_data = np.array([[dep_dim_samples[idx, 0], dep_dim_samples[idx, 1], cat_dim_samples[idx]] for idx in range(n_samples)])

######################################### Fitting the conical frustum ##################################################
elif fitting_mode == "conical_frustum":

    def estimate_ellipse_radii(feature_data, orientation, ref, n_ref_points):
        """Estimate the radii of the ellipse given the data and orientation.
        Args:
            feature_data:   Data into which the ellipse should be fitted
            orientation:    Orientation of the cylinder
            ref:            Reference vectors
            n_ref_points:   Number of data points that should be used to estimate the two radii
        Return:
            estimated_ellipse_radii: Radii of the estimated ellipse
        """
        feature_squarred = np.square(feature_data)

        if orientation == 0:
            indices = np.array([np.argpartition(feature_squarred[:, i+1], n_ref_points)[:n_ref_points]  for i in range(2)])
        elif orientation == 1:
            indices = np.array([np.argpartition(feature_squarred[:, i], n_ref_points)[:n_ref_points]  for i in range(0,3,2)])
        else:
            indices = np.array([np.argpartition(feature_squarred[:, i], n_ref_points)[:n_ref_points]  for i in range(2)])

        # Estimate the radii of the ellipse by taking the mean of the computed radii
        estimated_ellipse_radii = np.array([np.mean([cylinder_radius(feature_data[idx], ref[orientation]) for idx in indices[dim]]) for dim in range(2)])

        return estimated_ellipse_radii[::-1]

    def cylinder_radius(point, ref):
        """Caluclate the radius of the cylinder given a single point on the cylinder.
        Args:
            point:  Point on the cylinder
            ref:    Reference vectors
        Return:
            Calucalated radius
        """
        return np.linalg.norm(np.cross(point, (point-ref))) / np.linalg.norm(ref)
        
    ### Estimate the parameters of the conical frustum from the embedding ###
        
    ref = np.eye(3) # Setting reference vectors

    # Determine the orientation of the conical frustum
    radius_est = [[cylinder_radius(feature, r) for feature in emb_data] for r in ref]
    rad_mean = np.mean(radius_est, axis=1)
    rad_var = np.var(radius_est, axis=1)
    orientation = np.argmin(rad_var)
    
    # Set the number of latent points used for parameter estimation of the base sides
    amount_ref = 0.13
    n_ref_points = int(emb_data.shape[0] * amount_ref)
   
    # Select data points which are located at the ends of the frustum
    indices = np.array([np.argpartition(emb_data[:, orientation], n_ref_points)[:n_ref_points], np.argpartition(-emb_data[:, orientation], n_ref_points)[:n_ref_points]])
    sel_data_points = [[emb_data[idx] for idx in indices[i]] for i in range(2)]
    
    # Compute the radii of the bases
    estimated_radii_start = estimate_ellipse_radii(sel_data_points[0], orientation, ref, int(n_ref_points*amount_ref))
    estimated_radii_end = estimate_ellipse_radii(sel_data_points[1], orientation, ref, int(n_ref_points*amount_ref))

    # Determine the height of the frustum
    height_frustum = [np.amin(emb_data[:, orientation]), np.amax(emb_data[:, orientation])]


    ### Draw samples using the estimated parameters ###
    
    # Draw uniform samples along the height of the conical frustum
    height_feature = np.random.uniform(height_frustum[0], height_frustum[1], n_samples)
    
    # Compute radii in dependencie to the height feature
    slope = estimated_radii_end - estimated_radii_start # Compute slope between the bases
    height_feature_percentage = [(height_feature[i] - height_frustum[0]) / (height_frustum[1]-height_frustum[0]) for i in range(n_samples)]
    estimated_radii = np.array([height_feature_percentage[i] * slope + estimated_radii_start for i in range(n_samples)])

    # Draw samples from an ellipse with the estimated radii for each sample
    angle_feature = np.random.uniform(-np.pi, np.pi, n_samples)
    x_coor = np.cos(angle_feature) * estimated_radii[:, 0]
    y_coor = np.sin(angle_feature) * estimated_radii[:, 1]

    # Combine the drawn samples to latent points
    if orientation == 0:
        fitted_data = np.array([[height_feature[i], x_coor[i], y_coor[i]] for i in range(n_samples)])
    elif orientation == 1:
        fitted_data = np.array([[x_coor[i], height_feature[i], y_coor[i]] for i in range(n_samples)])
    else:
        fitted_data = np.array([[x_coor[i], y_coor[i], height_feature[i]] for i in range(n_samples)])
            
########################################################################################################################
# Perfom inverse ICA rotation to transform the embedding and fitted data back into the original coordinate system
emb_data = ica.inverse_transform(emb_data)
fitted_data = ica.inverse_transform(fitted_data)

######################################### Encoder-Decoder model ########################################################
log("Configuring the Decoder.")
decoder = Encoder_Decoder(n_features).decoder

print(decoder.summary())

log("Loading stored Decoder model weights.")
if fitting_mode is 'rectangular_frustum':
    decoder.load_weights('pretrained_models/fitting_prior_distributions/decoder_weights_rectangular.h5')
else:
    decoder.load_weights('pretrained_models/fitting_prior_distributions/decoder_weights_conical.h5')

########################################################################################################################
if plot_emb_recon:
    plotter = Plotter() # Instantiate plotter

    # Plot embedding and samples draw from the fitted frustum:
    plotter.plot_embedding_fitting(emb_data, fitted_data)

    # Computing and plotting decoder predictions
    predictions = decoder.predict(fitted_data, batch_size=fitted_data.shape[0])
    plotter.plot_images(predictions[:20])

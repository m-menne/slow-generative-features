########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file contains the code for plotting images, features maps, embeddings and predictions.
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
    
# Fixing seeds for comparison
numpy_seed = 0
np.random.seed(numpy_seed)

class Plotter():
    """Class for plotting input images, reconstruction images, feature maps, embeddings and predictions.
    """

    def __init__(self):
        pass
            
    def plot_recon_learning_curve_comp_ed_srae_wae(self, reconstruction_loss_ed, reconstruction_loss_srae, reconstruction_loss_wae, path=None):
        """Plot reconstruction losses of the Encoder-Decoder, SRAE and WAE model in comparison.
        Args:
            reconstruction_loss_ed:     Reconstruction loss of the Encoder-Decoder model
            reconstruction_loss_srae:   Reconstruction loss of the SRAE model
            reconstruction_loss_wae:    Reconstruction loss of the WAE model
            path:                       Path to store the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(20,12))
        
        ax.plot(reconstruction_loss_ed, label='Encoder-Decoder model')
        ax.plot(reconstruction_loss_srae, label=r'SRAE model ($\alpha = 15$)')
        ax.plot(reconstruction_loss_wae, label= r'Autoencoder')
        
        ax.set_ylim(0,0.12)
        ax.set(xlabel='Number of epochs', ylabel='Reconstruction loss')
        ax.legend()
        if path is not None:
            plt.savefig(path, format='eps', bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_recon_comp_ed_srae_wae(self, input_images, reconstruction_ed, reconstruction_srae, reconstruction_wae, path=None):
        """Plot the reconstructions of the Encoder-Decoder, SRAE and WAE model in comparison.
        Args:
            input_images:           List of input images
            reconstruction_ed:      List of reconstructions computed by the Encoder-Decoder model
            reconstruction_srae:    List of reconstructions computed by the SRAE model
            reconstruction_wae:     List of reconstructions computed by the WAE model
            path:                   Path to store the plot
        """
        fig, axs = plt.subplots(4, 10, figsize=(10,4))

        for i in range(10):
            axs[0, i].imshow(input_images[i].reshape(64,64), cmap='gray')
            axs[1, i].imshow(reconstruction_ed[i].reshape(64,64), cmap='gray')
            axs[2, i].imshow(reconstruction_srae[i].reshape(64,64), cmap='gray')
            axs[3, i].imshow(reconstruction_wae[i].reshape(64,64), cmap='gray')

        axs[0,0].set_ylabel('Input data', rotation=0, labelpad=200, ha='left', va='center')
        axs[1,0].set_ylabel('Reconstructed data\nEncoder-Decoder', rotation=0, labelpad=200, ha='left', va='center')
        axs[2,0].set_ylabel('Reconstructed data\nSRAE ' +r'($\alpha = 15$)', rotation=0, labelpad=200, ha='left', va='center')
        axs[3,0].set_ylabel('Reconstructed data\nAutoencoder', rotation=0, labelpad=200, ha='left', va='center')

        for ax in axs.flat:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            
        fig.subplots_adjust(wspace=0.04, hspace=0.04)
        if path is not None:
            plt.savefig(path, format='eps', bbox_inches='tight')
        plt.show()
        plt.close()
        
    def plot_latent_space_exploration(self, feature_values, path=None):
        """Plot the passed feature values according to their assigned class.
        Args:
            feature_values:     Array containing the feature values, assigned labels and confidence
            path:               Path to store the plot
        """
        from matplotlib import cm
        from matplotlib.colors import ListedColormap

        # Load Tab10 colormap
        tab10 = cm.get_cmap('tab10', 10)
        colors = [tab10(i) for i in range(10)]
        tab10 = [None] * 10
        N = 256
        for i in range(10):
           vals = np.ones((N, 4))
           vals[:, 0] = np.linspace(colors[i][0], 1, N)
           vals[:, 1] = np.linspace(colors[i][1], 1, N)
           vals[:, 2] = np.linspace(colors[i][2], 1, N)
           tab10[i] = ListedColormap(vals)
           
        # Load Set1 colormap
        set1 = cm.get_cmap('Set1', 9)
        colors = [set1(i) for i in range(9)]
        set1 = [None] * 9
        N = 256
        for i in range(9):
           vals = np.ones((N, 4))
           vals[:, 0] = np.linspace(colors[i][0], 1, N)
           vals[:, 1] = np.linspace(colors[i][1], 1, N)
           vals[:, 2] = np.linspace(colors[i][2], 1, N)
           set1[i] = ListedColormap(vals)

        # Set up colormap
        cmap_list = ['Purples_r', tab10[8], set1[6], tab10[0], 'Reds_r', 'Oranges_r', tab10[6], 'Greens_r', 'Greys_r', 'Blues_r']

        # Plot feature map
        fig, ax = plt.subplots(figsize=(20,12))
        for label in np.arange(10):
            current_label = np.array([feature for feature in feature_values if feature[2] == label])
            c = current_label[:,-1]
            if len(current_label) is not 0:
                scatter = ax.scatter(current_label[:,0], current_label[:,1], marker=".", c=c, cmap=cmap_list[label])

        ax.set(xlabel="1. Feature", ylabel="2. Feature")
        if path is not None:
            plt.savefig(path, format='png', bbox_inches='tight')
        plt.show()
        plt.close()
        
    def plot_embedding(self, items, embedded_data, attr_values, visualize_attr=0):
        """Plot embedding in 3D-scatter plot.
        Args:
            items:              Input data stored in list of items
            embedded_data:      Computed embedded data
            attr_values:        List of all possible attribute values
            visualize_attr:     Index of attribute to color-code
        """
        import matplotlib.colors as mplc
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
        
        c_value = np.arange(0.0, 1.0, 1.0/float(len(attr_values)))
        
        cmap = [mplc.to_rgb(cm.gist_rainbow(c_value[attr_values.index(item.attributes[visualize_attr])])) for item in items]

        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(111, projection='3d')

        ax1.scatter(embedded_data[:, 0],
                    embedded_data[:, 1],
                    embedded_data[:, 2],
                    c=cmap,
                    marker=".")

        # Hide grid lines
        ax1.grid(False)
        plt.axis('off')

        # Hide axes ticks
        ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])

        # Find optima
        optima = embedded_data.min(), embedded_data.max()
        ax1.set_xlim(*optima); ax1.set_ylim(*optima); ax1.set_zlim(*optima)
        
        plt.show()
        plt.close()
        
    def plot_comp_input_reconstruction(self, input_images, reconstructed_images, path=None):
        """Plotting 10 input images and the corresponding reconstructions.
        Args:
            input_images:           Array of input images
            reconstructed_images:   Array of reconstructed images
            path:                   Path to store the create figure
        """
        fig, axs = plt.subplots(2, 10, figsize=(10,2))
        
        for i in range(10):
            axs[0, i].imshow(input_images[i].reshape(64,64), cmap='gray')
            axs[1, i].imshow(reconstructed_images[i].reshape(64,64), cmap='gray')
        
        axs[0,0].set_ylabel('Input data', rotation=0, labelpad=200, ha='left', va='center')
        axs[1,0].set_ylabel('Reconstructed data', rotation=0, labelpad=200, ha='left', va='center')
        
        for ax in axs.flat:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            
        fig.subplots_adjust(wspace=0.04, hspace=0.04)
        if path is not None:
            plt.savefig(path, format='eps', bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_embedding_fitting(self, emb_data, fitted_data):
        """Plot embedding and samples drawn from the fitted frustum.
        Args:
            emb_data:       Embedded data
            fitted_data:    Samples drawn from the fitted frustum
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(emb_data[:, 0], emb_data[:, 1], emb_data[:, 2], marker=".", color='orange')
        ax.scatter(fitted_data[:, 0], fitted_data[:, 1], fitted_data[:, 2], marker=".")
                
        # Hide grid lines
        ax.grid(False)
        plt.axis('off')

        # Hide axes ticks
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

        # Find optima
        optima = emb_data.min(), emb_data.max()
        ax.set_xlim(*optima); ax.set_ylim(*optima); ax.set_zlim(*optima)

        plt.show()
        plt.close()
        
    def plot_images(self, input_images, path=None):
        """Plot 20 input images.
        Args:
            input_images:   Array of input images
        """
        fig, axs = plt.subplots(2, 10, figsize=(10,2))
        
        for i in range(2):
            for j in range(10):
                axs[i, j].imshow(input_images[i*10+j].reshape(64,64), cmap='gray')
            
        for ax in axs.flat:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            
        fig.subplots_adjust(wspace=0.04, hspace=0.04)
        if path is not None:
            plt.savefig(path, format='eps', bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_comp_input_ground_prediction(self, input_images, predicted_images, path=None):
        """Plot input, ground truth and predicted images in comparison.
        Args:
            input_images:       List of input and ground truth images
            predicted_images:   List of predicted images
        """
        plt.rcParams.update({'font.size': 18})
        
        fig, axs = plt.subplots(3, 10, figsize=(10,3))

        for i in range(10):
            axs[0, i].imshow(input_images[i].reshape(64,64), cmap='gray')
            axs[1, i].imshow(input_images[i+10].reshape(64,64), cmap='gray')
            axs[2, i].imshow(predicted_images[i+10].reshape(64,64), cmap='gray')

        axs[0,0].set_ylabel('Input data', rotation=0, labelpad=200, ha='left', va='center')
        axs[1,0].set_ylabel('Ground truth data', rotation=0, labelpad=200, ha='left', va='center')
        axs[2,0].set_ylabel('Predicted data', rotation=0, labelpad=200, ha='left', va='center')

        for ax in axs.flat:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            
        fig.subplots_adjust(wspace=0.04, hspace=0.04)
        if path is not None:
            plt.savefig(path, format='eps', bbox_inches='tight')
        plt.show()
        plt.close()

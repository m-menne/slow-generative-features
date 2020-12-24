########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file provides the implementations of the Encoder-Decoder model, Slowness-Regularized Autoencoder model,
# Autoencoder model, What-Where Encoder-Decoder model as well as the Encoder-Predictor-Decoder model.
########################################################################################################################

from core.powersfa import PowerWhitening

from keras.layers import *
from keras.models import Model

class Encoder_Decoder:
    """Encoder-Decoder model.
    """
    def __init__(self, n_features: int, n_power_iterations: int=250, distinguish_phases: bool=False):
        self.n_features = n_features
        self.n_power_iterations = n_power_iterations
        self.distinguish_phases = distinguish_phases
        self._build()
        
    def _build(self):
        # Encoder
        input = Input(shape=(64,64,1), name='input')
        x = Flatten()(input)
        x = Dense(self.n_features)(x)
        latent_features = PowerWhitening(output_dim=self.n_features, n_iterations=self.n_power_iterations, distinguish_phases=self.distinguish_phases, name='latent_feature')(x)

        # Decoder
        latent_input_features = Input(shape=(self.n_features,), name='latent_input')
        x = Dense(64, activation='relu')(latent_input_features)
        x = Dense(128, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(4096, activation='sigmoid')(x)
        reconstruction = Reshape((64,64,1), name='reconstruction')(x)

        self.encoder = Model(inputs=input, outputs=latent_features, name='Encoder model')
        self.decoder = Model(inputs=latent_input_features, outputs=reconstruction, name='Decoder model')
        
        return self.encoder, self.decoder

class SRAE:
    """Slowness-Regularized Autoencoder model.
    """
    def __init__(self, n_features: int, n_power_iterations: int=250, distinguish_phases: bool=False):
        self.n_features = n_features
        self.n_power_iterations = n_power_iterations
        self.distinguish_phases = distinguish_phases
        self._build()
        
    def _build(self):
        # Build SRAE model
        # Encoder
        input = Input(shape=(64,64,1), name='input')
        x = Flatten()(input)
        x = Dense(self.n_features)(x)
        latent_features = PowerWhitening(output_dim=self.n_features, n_iterations=self.n_power_iterations, distinguish_phases=self.distinguish_phases, name='latent_feature')(x)
        
        # Define Decoder layers
        self.Dense1 = Dense(64, activation='relu')
        self.Dense2 = Dense(128, activation='relu')
        self.Dense3 = Dense(256, activation='relu')
        self.Dense4 = Dense(512, activation='relu')
        self.Dense5 = Dense(4096, activation='sigmoid')
        
        # Build Decoder
        x = self.Dense1(latent_features)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)
        x = self.Dense5(x)
        output = Reshape((64,64,1), name='reconstruction')(x)
        
        # Build pure Decoder model
        latent_input_features = Input(shape=(self.n_features,), name='latent_input')
        x = self.Dense1(latent_input_features)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)
        x = self.Dense5(x)
        reconstruction = Reshape((64,64,1), name='reconstruction')(x)

        self.model = Model(inputs=input, outputs=[latent_features, output], name='SRAE model')
        self.encoder = Model(inputs=input, outputs=latent_features, name='Encoder model')
        self.decoder = Model(inputs=latent_input_features, outputs=reconstruction, name='Decoder model')
        
        return self.model, self.encoder, self.decoder
        
class AE:
    """Autoencoder model.
    """
    def __init__(self, n_features: int):
        self.n_features = n_features
        self._build()
        
    def _build(self):
        # Build AE model
        # Encoder
        input = Input(shape=(64,64,1), name='input')
        x = Flatten()(input)
        latent_features = Dense(self.n_features)(x)
        
        # Define Decoder layers
        self.Dense1 = Dense(64, activation='relu')
        self.Dense2 = Dense(128, activation='relu')
        self.Dense3 = Dense(256, activation='relu')
        self.Dense4 = Dense(512, activation='relu')
        self.Dense5 = Dense(4096, activation='sigmoid')
        
        # Build Decoder
        x = self.Dense1(latent_features)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)
        x = self.Dense5(x)
        output = Reshape((64,64,1), name='reconstruction')(x)
        
        # Build pure Decoder model
        latent_input_features = Input(shape=(self.n_features,), name='latent_input')
        x = self.Dense1(latent_input_features)
        x = self.Dense2(x)
        x = self.Dense3(x)
        x = self.Dense4(x)
        x = self.Dense5(x)
        reconstruction = Reshape((64,64,1), name='reconstruction')(x)

        self.model = Model(inputs=input, outputs=output, name='AE model')
        self.encoder = Model(inputs=input, outputs=latent_features, name='Encoder model')
        self.decoder = Model(inputs=latent_input_features, outputs=reconstruction, name='Decoder model')
        
        return self.model, self.encoder, self.decoder

class What_Where_Encoder_Decoder:
    """What-Where Encoder-Decoder model.
    """
    def __init__(self, n_enc_0_features: int, n_enc_1_features: int, return_enc: int, n_power_iterations: int=250, distinguish_phases: bool=False):
        self.n_enc_0_features = n_enc_0_features
        self.n_enc_1_features = n_enc_1_features
        self.return_enc = return_enc
        self.n_power_iterations = n_power_iterations
        self.distinguish_phases = distinguish_phases
        self._build()
        
    def _build(self):
        if self.return_enc is 0:
            # Encoder 0
            input_0 = Input(shape=(64,64,1), name='input')
            x = Flatten()(input_0)
            x = Dense(self.n_enc_0_features)(x)
            latent_features_0 = PowerWhitening(output_dim=self.n_enc_0_features, n_iterations=self.n_power_iterations, distinguish_phases=self.distinguish_phases, name='latent_feature_0')(x)
        else:
            # Encoder 1
            input_1 = Input(shape=(64,64,1), name='input')
            x = Flatten()(input_1)
            x = Dense(self.n_enc_1_features)(x)
            latent_features_1 = PowerWhitening(output_dim=self.n_enc_1_features, n_iterations=self.n_power_iterations, distinguish_phases=self.distinguish_phases, name='latent_feature_1')(x)

        # Decoder
        latent_input_features = Input(shape=(self.n_enc_0_features + self.n_enc_1_features,), name='latent_input')
        x = Dense(64, activation='relu')(latent_input_features)
        x = Dense(128, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(4096, activation='sigmoid')(x)
        reconstruction = Reshape((64,64,1), name='reconstruction')(x)

        if self.return_enc is 0:
            self.encoder = Model(inputs=input_0, outputs=latent_features_0, name='Encoder_0 model')
        else:
            self.encoder = Model(inputs=input_1, outputs=latent_features_1, name='Encoder_1 model')
        self.decoder = Model(inputs=latent_input_features, outputs=reconstruction, name='Decoder model')
        
        return self.encoder, self.decoder

class Encoder_Predictor_Decoder:
    """Encoder-Predictor-Decoder model.
    """
    def __init__(self, n_features: int, seq_length_in: int, seq_length_out: int, n_power_iterations: int=250, distinguish_phases: bool=False):
        self.n_features = n_features
        self.seq_length_in = seq_length_in
        self.seq_length_out = seq_length_out
        self.n_power_iterations = n_power_iterations
        self.distinguish_phases = distinguish_phases
        self._build()
        
    def _build(self):
        # Encoder
        input = Input(shape=(64,64,1), name='input')
        x = Flatten()(input)
        x = Dense(self.n_features)(x)
        latent_features = PowerWhitening(output_dim=self.n_features, n_iterations=self.n_power_iterations, distinguish_phases=self.distinguish_phases, name='latent_feature')(x)

        # Predictor
        input_history = Input(shape=(self.seq_length_in, self.n_features), name='history_input')
        x = LSTM(64, return_sequences=True)(input_history)
        x = LSTM(32, activation='relu')(x)
        x = Dense(self.seq_length_out * self.n_features)(x)
        prediction = Reshape((self.seq_length_out, self.n_features), name='prediction')(x)

        # Decoder
        latent_input_features = Input(shape=(self.n_features,), name='latent_input')
        x = Dense(64, activation='relu')(latent_input_features)
        x = Dense(128, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(4096, activation='sigmoid')(x)
        reconstruction = Reshape((64,64,1), name='reconstruction')(x)

        self.encoder = Model(inputs=input, outputs=latent_features, name='Encoder model')
        self.predictor = Model(inputs=input_history, outputs=prediction, name='Prediction model')
        self.decoder = Model(inputs=latent_input_features, outputs=reconstruction, name='Decoder model')
        
        return self.encoder, self.predictor, self.decoder



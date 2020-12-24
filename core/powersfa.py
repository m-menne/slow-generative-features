########################################################################################################################
# This code is for the paper "Exploring Slow Feature Analysis for Generative Latent Factors"
# by Max Menne, Merlin Schüler, and Laurenz Wiskott to be published at ICPRAM 2021.
#
# This file contains the code for PowerSFA framework. The code is adapted from the paper
# "Gradient-based Training of Slow Feature Analysis by Differentiable Approximate Whitening"
# by Merlin Schüler, Hlynur Davíð Hlynsson and Laurenz Wiskott:
# http://proceedings.mlr.press/v101/schuler19a.html
########################################################################################################################

import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras
import keras.backend as K
from keras.layers import *
from keras.engine.topology import Layer
from keras.backend.tensorflow_backend import set_session

import numpy as np

def sfa_adjacency_matrix(sfa_batch_size):
    """ 
    This function computes a weight matrix for emulating mini-batch slowness optimization
    with a generalized SFA method. It assumes the batches to be composed of a number of 
    single transitions.

    """
    W = np.zeros((sfa_batch_size, sfa_batch_size))
    for i in range(sfa_batch_size):
        for j in range(sfa_batch_size):
            if abs(i-j) < 2 and i<j and i%2 is 0 and j%2 is 1:
                W[i, j] = 1
                W[j, i] = 1
    return W/(2 * sfa_batch_size)

def matrix_power(A, x, power):
    """
    This Tensorflow function repeatedly applies a matrix to a vector
    and normalizes the result. 
    """
    def single_power_step(A, x):
        x = tf.matmul(A, x)
        x = tf.math.divide(x, tf.norm(x))
        return x
    iter_count_tf = tf.constant(0)
    condition  = lambda it, A, x: tf.less(it, power)
    body = lambda it, A, x: (it+1, A, single_power_step(A, x))
    loop_vars = [iter_count_tf, A, x]
    output = tf.while_loop(condition, body, loop_vars)[2]
    e = tf.norm(tf.matmul(A, output))
    return output, e

def unnormalized_laplacian(weight_matrix_tf):
    """
    (Tensorflow)
    Provides a computation graph for the unnormalized Laplacian matrix as L = D - W
    from the weight matrix W.
    :param weight_matrix_tf: the tensor for the weight matrix W - should be real & symmetric
    :return: tensor for the unnormalized Laplacian matrix, shape=(n_data, n_data).
    """
    degree_vector_tf = tf.reduce_sum(weight_matrix_tf, 0)
    degree_matrix_tf = tf.linalg.tensor_diag(degree_vector_tf)
    laplacian_tf = degree_matrix_tf - weight_matrix_tf
    return laplacian_tf

def normalized_laplacian(weight_matrix_tf):
    """
    (Tensorflow)
    Provides a computation graph for the normalized Laplacian matrix as L = I - D^(-1/2)WD^(-1/2)
    from the weight matrix W.
    :param weight_matrix_tf: the tensor for the weight matrix W - should be real & symmetric
    :return: tensor for the normalized Laplacian matrix, shape=(n_data, n_data).
    """
    degree_vector_tf = tf.reduce_sum(weight_matrix_tf, 0)
    degree_matrix_tf = tf.diag(degree_vector_tf)
    laplacian_tf = degree_matrix_tf - weight_matrix_tf
    sqrt_inv_degree_matrix_tf = tf.diag(1./tf.sqrt(degree_vector_tf))
    normalized_laplacian_tf = tf.matmul(sqrt_inv_degree_matrix_tf, tf.matmul(laplacian_tf, sqrt_inv_degree_matrix_tf))
    return normalized_laplacian_tf

def unit_variance(input_tensor):
    input_mean, _ = tf.nn.moments(input_tensor, axes=[0])
    input_tensor = input_tensor - input_mean[None, :]
    input_std = tf.sqrt(tf.reduce_mean(tf.square(input_tensor), axis=0))
    input_tensor = input_tensor / input_std
    return input_tensor

def ordered_gsfa_loss(yTrue, yPred, normalize_laplacian=False):
    """
    This loss function implements the generalized SFA loss and tries
    to enforce ordering in the output features by weighting the contributions
    of every single feature in a monotonically decreasing fashion. 
    Slowness loss can be emulated by using a corresponding weight matrix.
    
    (Hacky to comply with keras. yTrue actually contains weight matrix.)
    """
    weight_matrix_tf = yTrue
    input_tensor = yPred
    output_dim = input_tensor.shape[1]
    feature_weight = tf.linspace(1., 0.3, output_dim)
    feature_weight *= 1./np.linalg.norm(np.linspace(1., 0.3, output_dim))
    if normalize_laplacian:
        laplacian_tf = normalized_laplacian(weight_matrix_tf)
    else:
        laplacian_tf = unnormalized_laplacian(weight_matrix_tf)
    weighted_whitened_output = input_tensor * feature_weight
    auxiliary_matrix = tf.matmul(weighted_whitened_output, tf.matmul(laplacian_tf, weighted_whitened_output), True)
    auxiliary_diagonal = tf.diag_part(auxiliary_matrix)
    loss = tf.reduce_mean(auxiliary_diagonal) # reduces to mean over features (not over number of steps!# )
    return loss

def unordered_gsfa_loss(yTrue, yPred, normalize_laplacian=False):
    """
    This loss function implements the generalized SFA loss with unordered features.
    Slowness loss can be emulated by using a corresponding weight matrix.
    
    (Hacky to comply with keras. yTrue actually contains weight matrix.)
    """
    weight_matrix_tf = yTrue
    input_tensor = yPred
    output_dim = input_tensor.shape[1]
    feature_weight = tf.constant(1.)
    if normalize_laplacian:
        laplacian_tf = normalized_laplacian(weight_matrix_tf)
    else:
        laplacian_tf = unnormalized_laplacian(weight_matrix_tf)
    weighted_whitened_output = input_tensor * feature_weight
    auxiliary_matrix = tf.matmul(weighted_whitened_output, tf.matmul(laplacian_tf, weighted_whitened_output), True)
    auxiliary_diagonal = tf.linalg.tensor_diag_part(auxiliary_matrix)
    loss = tf.reduce_mean(auxiliary_diagonal) # reduces to mean over features (not over number of steps!# )
    return loss

def compute_whitening_matrix_tf(covariance_matrix, output_dim, n_iterations=50, layer_id=0, random_init=None):
    #with tf.name_scope("whitening_matrix"):

    if random_init is None:
        initializer = np.random.normal(size=(output_dim, output_dim))
    else:
        assert(random_init.shape == (output_dim, output_dim))
        initializer = random_init
    R = tf.compat.v1.get_variable("random_vectors"+str(layer_id),
                        initializer= initializer.astype(np.float32),
                        trainable = False)
    whitening_matrix = tf.compat.v1.get_variable("whiteningmatrix"+str(layer_id),
                               initializer=np.zeros(shape=(output_dim, output_dim)).astype(np.float32),
                               trainable=False)
    init_custom_op = tf.compat.v1.variables_initializer(var_list=[whitening_matrix, R])
    with tf.control_dependencies([init_custom_op]):
        iter_count_tf = tf.constant(0)
        condition = lambda it, covariance_matrix, W, R: tf.less(it, output_dim)
        def body(it, covariance_matrix, W, R):
            v, l = matrix_power(covariance_matrix, R[:, it, None], n_iterations)
            return (it+1,
                    covariance_matrix - l * tf.matmul(v, v, False, True),
                    W + 1 / tf.sqrt(l) * tf.matmul(v, v, False, True),
                    R)
        whitening_matrix = tf.while_loop(condition,
                                 body,
                                 [iter_count_tf, covariance_matrix, whitening_matrix, R])[2]
    return whitening_matrix

class SFABatchGenerator():
    """
    A batch generator to use with Keras' fit_generator method. Will take (ordered)
    time-series data and generate a batch of random transition samples.
    """
    def __init__(self, data, batch_size, batch_replacement=False, frameskip=0):
        self.data = data
        self.frameskip = frameskip
        self.batch_replacement = batch_replacement
        self.batch_size = batch_size
        self.index_mask = np.ones(data.shape[0] - 1 - frameskip, bool)
        self.all_indices = np.arange(0, self.data.shape[0] - 1 - frameskip, 1)
        self.max_iter = len(self)
        self.current_iter = 0
        self.adjacency_matrix = sfa_adjacency_matrix(self.batch_size)
        assert(batch_size % 2 == 0)

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.data.shape[0]/self.batch_size)

    def __next__(self):
        self.current_iter += 1
        batch_indices = np.zeros((self.batch_size,), dtype=np.uint32)
        batch_indices[::2] = np.random.choice(self.all_indices[self.index_mask],
                                              int(self.batch_size / 2),
                                              replace=self.batch_replacement)
        if not self.batch_replacement:
            self.index_mask[batch_indices[::2]] = False
        if self.index_mask.sum() < self.batch_size:
            self.index_mask = np.ones(self.data.shape[0] - 1 - self.frameskip, bool)
        batch_indices[1::2] = batch_indices[::2] + 1 + self.frameskip
        return self.data[batch_indices], self.adjacency_matrix

class GeneralizedBatchGenerator():
    """
    A batch generator to use with Keras' fit_generator method. Will take a dataset
    and a corresponding similarity matrix and generates batches of random samples as
    well as the corresponding similarity submatrices.
    """
    def __init__(self, data, W, batch_size, batch_replacement=False, connection_dropout=0, connection_dropin=1):
        self.data = data
        self.batch_replacement = batch_replacement
        self.batch_size = batch_size
        self.index_mask = np.ones(data.shape[0] - 1, bool)
        self.all_indices = np.arange(0, self.data.shape[0] - 1, 1)
        self.max_iter = len(self)
        self.current_iter = 0
        self.adjacency_matrix = W
        assert(connection_dropout >= 0)
        assert(connection_dropout <= 1)
        self.connection_dropout = connection_dropout
        assert(connection_dropin >= 1)
        self.connection_dropin = connection_dropin

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.data.shape[0] / self.batch_size)

    def __next__(self):
        self.current_iter += 1
        batch_indices = np.random.choice(self.all_indices[self.index_mask], self.batch_size,
                                              replace=self.batch_replacement)
        if not self.batch_replacement:
            self.index_mask[batch_indices] = False
            if self.index_mask.sum() < self.batch_size:
                self.index_mask = np.ones(self.data.shape[0] - 1, bool)
        sub_W = self.adjacency_matrix[np.ix_(batch_indices, batch_indices)]
        if (self.connection_dropout > 0):
            dropout_mask = np.random.uniform(0, 1, size=sub_W.shape) > self.connection_dropout
            sub_W *= dropout_mask
        if (self.connection_dropin > 1):
            dropin_mask = np.random.uniform(1, self.connection_dropin, size=sub_W.shape)
            sub_W *= dropin_mask 
        return self.data[batch_indices], sub_W
        
class GeneralizedBatchGenerator2():
    """
    A batch generator to use with Keras' fit_generator method. Will take a dataset
    and a corresponding similarity matrix and generates batches of random samples as
    well as the corresponding similarity submatrices. Returns random samples as input
    and output as well as the corresponding similarity submatrices.
    """
    def __init__(self, data, W, batch_size, batch_replacement=False, connection_dropout=0, connection_dropin=1):
        self.data = data
        self.batch_replacement = batch_replacement
        self.batch_size = batch_size
        self.index_mask = np.ones(data.shape[0] - 1, bool)
        self.all_indices = np.arange(0, self.data.shape[0] - 1, 1)
        self.max_iter = len(self)
        self.current_iter = 0
        self.adjacency_matrix = W
        assert(connection_dropout >= 0)
        assert(connection_dropout <= 1)
        self.connection_dropout = connection_dropout
        assert(connection_dropin >= 1)
        self.connection_dropin = connection_dropin

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.data.shape[0] / self.batch_size)

    def __next__(self):
        self.current_iter += 1
        batch_indices = np.random.choice(self.all_indices[self.index_mask], self.batch_size,
                                              replace=self.batch_replacement)
        if not self.batch_replacement:
            self.index_mask[batch_indices] = False
            if self.index_mask.sum() < self.batch_size:
                self.index_mask = np.ones(self.data.shape[0] - 1, bool)
        sub_W = self.adjacency_matrix[np.ix_(batch_indices, batch_indices)]
        if (self.connection_dropout > 0):
            dropout_mask = np.random.uniform(0, 1, size=sub_W.shape) > self.connection_dropout
            sub_W *= dropout_mask
        if (self.connection_dropin > 1):
            dropin_mask = np.random.uniform(1, self.connection_dropin, size=sub_W.shape)
            sub_W *= dropin_mask
        return self.data[batch_indices], [sub_W, self.data[batch_indices]]
        
class GeneralizedBatchGenerator3():
    """
    A batch generator to use with Keras' fit_generator method. Will take a dataset
    and generates batches of random samples. Returns random samples as input
    and output.
    """
    def __init__(self, data, batch_size, batch_replacement=False):
        self.data = data
        self.batch_replacement = batch_replacement
        self.batch_size = batch_size
        self.index_mask = np.ones(data.shape[0] - 1, bool)
        self.all_indices = np.arange(0, self.data.shape[0] - 1, 1)
        self.max_iter = len(self)
        self.current_iter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.data.shape[0] / self.batch_size)

    def __next__(self):
        self.current_iter += 1
        batch_indices = np.random.choice(self.all_indices[self.index_mask], self.batch_size,
                                              replace=self.batch_replacement)
        
        return self.data[batch_indices], self.data[batch_indices]


class SwitchLayer(Layer):
    """
    An auxiliary layer to help power whitening to distinguish between training and test phase.
    """
    def __init__(self, **kwargs):
        super(SwitchLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(SwitchLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        output = K.in_train_phase(inputs[0], inputs[1])
        return output


class PowerWhitening(Layer):
    """
    A Keras' layer implementing the differentiable power whitening procedure.
    """
    def __init__(self, output_dim=1, n_iterations=100, mixing_coefficient=0.0, distinguish_phases=True, layer_id=0, random_init=None, **kwargs):
        self.output_dim = output_dim
        self.n_iterations = n_iterations
        if (distinguish_phases):
            self.phase_detector = None # this should later default to the backends learning phase indicator
        else:
            self.phase_detector = tf.constant(1)
        self.mixing_coefficient = mixing_coefficient
        self.last_covariance_matrix = None
        self.layer_id = layer_id
        #self.covariance_matrix = None
        self.last_mean = tf.compat.v1.get_variable("last_mean"+str(self.layer_id), (output_dim,), tf.float32, initializer=tf.initializers.zeros, trainable=False)
        self.last_whitening_matrix = tf.compat.v1.get_variable("last_whitening_matrix"+str(self.layer_id),
                                                 (output_dim, output_dim),
                                                 tf.float32,
                                                 initializer=tf.initializers.identity,
                                                 trainable=False)
        if random_init is None:
            self.random_vectors_init = np.random.normal(0, 1, (output_dim, output_dim))
        else:
            assert (random_init.shape == (output_dim, output_dim))
            self.random_vectors_init = random_init
        self. _uses_learning_phase = True
        super(PowerWhitening, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PowerWhitening, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, input_tensor, layer_id=0, mask=None):
        mean_lambda_layer = Lambda(lambda inp: K.in_train_phase(inp[0], inp[1], self.phase_detector)) 
        input_mean = mean_lambda_layer([tf.nn.moments(input_tensor, axes=[0])[0], self.last_mean])
        mixed_mean = self.mixing_coefficient * self.last_mean + (1 - self.mixing_coefficient) * input_mean
        mean_lambda_layer._uses_learning_phase = True
        train_input_tensor = input_tensor - input_mean[None, :]
        self.covariance_matrix = tf.math.divide(tf.matmul(train_input_tensor, train_input_tensor, True, False), tf.cast(tf.shape(train_input_tensor)[0], tf.float32))
        self.last_covariance_matrix = tf.compat.v1.get_variable("last_covariance_matrix"+str(self.layer_id),
                                                 self.covariance_matrix.shape,
                                                 tf.float32,
                                                 initializer=tf.initializers.identity,
                                                 trainable=False)
        mixed_covariance = self.mixing_coefficient * self.last_covariance_matrix + (1 - self.mixing_coefficient) * self.covariance_matrix
        mix_cov_lambda_layer = Lambda(lambda inp: K.in_train_phase(inp[0], inp[1], self.phase_detector)) 
        mixed_covariance = mix_cov_lambda_layer([mixed_covariance, self.last_covariance_matrix])
        mix_cov_lambda_layer._uses_learning_phase = True
        whitening_matrix = compute_whitening_matrix_tf(covariance_matrix=mixed_covariance,
                                                       output_dim=self.output_dim,
                                                       n_iterations=self.n_iterations,
                                                       layer_id=layer_id) 
        whitening_lambda_layer= Lambda(lambda inp: K.in_train_phase(inp[0], inp[1], self.phase_detector))
        whitening_matrix = whitening_lambda_layer([whitening_matrix, self.last_whitening_matrix])
        whitening_lambda_layer._uses_learning_phase = True
        with tf.control_dependencies([whitening_matrix]):
            assign_op_mean = tf.compat.v1.assign(self.last_mean, mixed_mean)
            assign_op_mixed = tf.compat.v1.assign(self.last_covariance_matrix, mixed_covariance)
            assign_op_whitening = tf.compat.v1.assign(self.last_whitening_matrix, whitening_matrix)
            with tf.control_dependencies([assign_op_mixed, assign_op_whitening,
                assign_op_mean]):
                whitening_matrix = tf.identity(whitening_matrix)
                whitened_output = tf.matmul(train_input_tensor, whitening_matrix, False, True) 
        test_phase_output = tf.matmul(input_tensor - self.last_mean[None, :], whitening_matrix, False, True)
        switch_layer = Lambda(lambda inp: K.in_train_phase(inp[0], inp[1], self.phase_detector))
        switch_layer._uses_learning_phase = True
        output = switch_layer([whitened_output, test_phase_output])
        return output

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements ESN Cell."""

import tensorflow as tf
import tensorflow.keras as keras
# from typeguard import typechecked
import numpy as np

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from FixedPointStore_leaky import FixedPointStore
from FixedPointSearch_leaky import FixedPointSearch
# from parameters import par
import sys
import config

sys.path.append('C:/Users/Nishant Joshi/Downloads/DDN-public/rnn_flip_flops/rate_nets/rate_TF_cell')

import matplotlib.pyplot as plt 
# from tensorflow_addons.utils.types import (
#     Activation,
#     Initializer,
# )
import os 

par  = {'units': 100,
        'connectivity': 0.1,
        'leaky': 1,
        'spectral_radius':  0.9,
        'use_norm2':  False,
        'use_bias':  True,
        'activation': "tanh",
}

@tf.keras.utils.register_keras_serializable(package="Addons")

class ESNCell(keras.layers.AbstractRNNCell):
    """Echo State recurrent Network (ESN) cell.
    This implements the recurrent cell from the paper:
        H. Jaeger
        "The "echo state" approach to analysing and training recurrent neural networks".
        GMD Report148, German National Research Center for Information Technology, 2001.
        https://www.researchgate.net/publication/215385037

    Example:

    >>> inputs = np.random.random([30,23,9]).astype(np.float32)
    >>> ESNCell = tfa.rnn.ESNCell(4)
    >>> rnn = tf.keras.layers.RNN(ESNCell, return_sequences=True, return_state=True)
    >>> outputs, memory_state = rnn(inputs)
    >>> outputs.shape
    TensorShape([30, 23, 4])
    >>> memory_state.shape
    TensorShape([30, 4])

    Args:
        units: Positive integer, dimensionality in the reservoir.
        connectivity: Float between 0 and 1.
            Connection probability between two reservoir units.
            Default: 0.1.
        leaky: Float between 0 and 1.
            Leaking rate of the reservoir.
            If you pass 1, it is the special case the model does not have leaky
            integration.
            Default: 1.
        spectral_radius: Float between 0 and 1.
            Desired spectral radius of recurrent weight matrix.
            Default: 0.9.
        use_norm2: Boolean, whether to use the p-norm function (with p=2) as an upper
            bound of the spectral radius so that the echo state property is satisfied.
            It  avoids to compute the eigenvalues which has an exponential complexity.
            Default: False.
        use_bias: Boolean, whether the layer uses a bias vector.
            Default: True.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            Default: `glorot_uniform`.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights matrix,
            used for the linear transformation of the recurrent state.
            Default: `glorot_uniform`.
        bias_initializer: Initializer for the bias vector.
            Default: `zeros`.
    Call arguments:
        inputs: A 2D tensor (batch x num_units).
        states: List of state tensors corresponding to the previous timestep.
    """

    # @typechecked
    def __init__(
        self,
        units: int,
        connectivity: float = 0.1,
        leaky: float = 1,
        spectral_radius: float = 0.9,
        use_norm2: bool = False,
        use_bias: bool = True,
        activation: str =  "tanh",
        kernel_initializer: str  = "glorot_uniform",
        recurrent_initializer:  str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.connectivity = connectivity
        self.leaky = leaky
        self.spectral_radius = spectral_radius
        self.use_norm2 = use_norm2
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._state_size = units
        self._output_size = units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        input_size = tf.compat.dimension_value(tf.TensorShape(inputs_shape)[-1])
        if input_size is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]. Shape received is %s"
                % inputs_shape
            )

        def _esn_recurrent_initializer(shape, dtype, partition_info=None):
            recurrent_weights = tf.keras.initializers.get(self.recurrent_initializer)(
                shape, dtype
            )

            connectivity_mask = tf.cast(
                tf.math.less_equal(tf.random.uniform(shape), self.connectivity),
                dtype,
            )
            recurrent_weights = tf.math.multiply(recurrent_weights, connectivity_mask)

            # Satisfy the necessary condition for the echo state property `max(eig(W)) < 1`
            if self.use_norm2:
                # This condition is approximated scaling the norm 2 of the reservoir matrix
                # which is an upper bound of the spectral radius.
                recurrent_norm2 = tf.math.sqrt(
                    tf.math.reduce_sum(tf.math.square(recurrent_weights))
                )
                is_norm2_0 = tf.cast(tf.math.equal(recurrent_norm2, 0), dtype)
                scaling_factor = tf.cast(self.spectral_radius, dtype) / (
                    recurrent_norm2 + 1 * is_norm2_0
                )
            else:
                abs_eig_values = tf.abs(tf.linalg.eig(recurrent_weights)[0])
                scaling_factor = tf.math.divide_no_nan(
                    tf.cast(self.spectral_radius, dtype), tf.reduce_max(abs_eig_values)
                )

            recurrent_weights = tf.multiply(recurrent_weights, scaling_factor)

            return recurrent_weights



        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=[self.units, self.units],
            initializer=_esn_recurrent_initializer,
            trainable=False,
            dtype=self.dtype,
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_size, self.units],
            initializer=self.kernel_initializer,
            trainable=False,
            dtype=self.dtype,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.units],
                initializer=self.bias_initializer,
                trainable=False,
                dtype=self.dtype,
            )

        self.built = True



    def call(self, inputs, state):
        in_matrix = tf.cast(tf.concat([inputs, state], axis=1),dtype='float32')
        weights_matrix = tf.concat([self.kernel, self.recurrent_kernel], axis=0)

        output = tf.linalg.matmul(in_matrix, weights_matrix)
        if self.use_bias:
            output = output + self.bias
        output = self.activation(output)
        output = (1 - self.leaky) * state[0] + self.leaky * output
        return output, output



    def get_config(self):
        config = {
            "units": self.units,
            "connectivity": self.connectivity,
            "leaky": self.leaky,
            "spectral_radius": self.spectral_radius,
            "use_norm2": self.use_norm2,
            "use_bias": self.use_bias,
            "activation": tf.keras.activations.serialize(self.activation),
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": tf.keras.initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

# Static functions

@tf.keras.utils.register_keras_serializable(package="Addons")

class ESNCell_S(keras.layers.AbstractRNNCell):
    """Echo State recurrent Network (ESN) cell.
    This implements the recurrent cell from the paper:
        H. Jaeger
        "The "echo state" approach to analysing and training recurrent neural networks".
        GMD Report148, German National Research Center for Information Technology, 2001.
        https://www.researchgate.net/publication/215385037

    Example:

    >>> inputs = np.random.random([30,23,9]).astype(np.float32)
    >>> ESNCell = tfa.rnn.ESNCell(4)
    >>> rnn = tf.keras.layers.RNN(ESNCell, return_sequences=True, return_state=True)
    >>> outputs, memory_state = rnn(inputs)
    >>> outputs.shape
    TensorShape([30, 23, 4])
    >>> memory_state.shape
    TensorShape([30, 4])

    Args:
        units: Positive integer, dimensionality in the reservoir.
        connectivity: Float between 0 and 1.
            Connection probability between two reservoir units.
            Default: 0.1.
        leaky: Float between 0 and 1.
            Leaking rate of the reservoir.
            If you pass 1, it is the special case the model does not have leaky
            integration.
            Default: 1.
        spectral_radius: Float between 0 and 1.
            Desired spectral radius of recurrent weight matrix.
            Default: 0.9.
        use_norm2: Boolean, whether to use the p-norm function (with p=2) as an upper
            bound of the spectral radius so that the echo state property is satisfied.
            It  avoids to compute the eigenvalues which has an exponential complexity.
            Default: False.
        use_bias: Boolean, whether the layer uses a bias vector.
            Default: True.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            Default: `glorot_uniform`.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights matrix,
            used for the linear transformation of the recurrent state.
            Default: `glorot_uniform`.
        bias_initializer: Initializer for the bias vector.
            Default: `zeros`.
    Call arguments:
        inputs: A 2D tensor (batch x num_units).
        states: List of state tensors corresponding to the previous timestep.
    """

    # @typechecked
    def __init__(self,
                weights: np.ndarray,
                bias: np.ndarray  ,
                n_type: np.ndarray ,
                coordinates:np.ndarray ,
                decay: np.ndarray,
                input_n: np.ndarray= np.array([0, 1, 2]), 
                output_n: np.ndarray = np.array([-3, -2, -1]),
                activation_func: str = None, 
                dt: float = 0.0005, 
                theta_window : np.ndarray = None, 
                theta_y0:  np.ndarray = None,
                lr: float = 1.0, 
                var_delays: bool = True, 

                **kwargs):
        
        
        super().__init__(**kwargs)
        self.x_range = (np.min(coordinates[:, 0]), np.max(coordinates[:, 0]))
        self.y_range = (np.min(coordinates[:, 1]), np.max(coordinates[:, 1]))
        self.spatial_dist_continuous = coordinates2distance(coordinates)


        # Discretized distance matrix according to given dt
        dist_per_step = dt * config.propagation_vel
        self.D = np.asarray(np.ceil(self.spatial_dist_continuous / dist_per_step), dtype='int32')

        longest_delay_needed = np.max(self.D) + 1

        if not var_delays:
            self.D = np.ones_like(self.spatial_dist_continuous)
            np.fill_diagonal(self.D, 0)

        self.coordinates = coordinates
        self.dt = dt
        self.N = coordinates.shape[0]  # Number of neurons
        self.W = weights
        self.WBias = bias
        self.n_type = tf.constant(n_type, dtype='float32')
        self.B = longest_delay_needed  # Buffer size
        self.A_init = np.zeros((self.N, self.B))
        self.A = np.copy(self.A_init)

        self.neurons_in = input_n  # Indices for input neurons
        self.neurons_out = output_n  # Indices for output neurons
        self.decay = decay
        self.weight_decay = 0.01
        if self.W is not None:
            self.lr = lr * np.array(self.W > 0, dtype='uint8')
        else:
            self.lr = lr
        self.theta = np.ones((self.N,))
        self.theta_window = theta_window
        self.theta_y0 = theta_y0
        if self.theta_window is None:
            self.theta_window = self.A.shape[1]
        assert self.theta_window <= self.A.shape[1], 'Window size for theta can not be larger than buffer size for ' \
                                                     'net activity. '

        if self.theta_y0 is None:
            self.theta_y0 = 1

        if activation_func is None:
            self.activation_func = tanh_activation_tf
        else:
            self.activation_func = activation_func

        self.mid_dist = (np.max(self.D) - 1) / 2

        # Compute masked weight matrices
        self.W_masked_list = [self.W]
        self.lr_masked_list = [self.lr]
        if not (self.W is None) and var_delays:
            self.compute_masked_W()
            self.compute_masked_lr()

        self.W_masked_list_init = [np.copy(partial_W) for partial_W in self.W_masked_list]

        self.var_delays = var_delays

        # compute connectivity matrix for use in structural plasticity
         
    @property
    def state_size(self):
        return self.N

    @property
    def output_size(self):
        return self.neurons_out


    def reset_weigths(self, W):
        self.W = W
        self.lr = self.lr * np.array(self.W > 0, dtype='uint8')
        self.W_masked_list = [self.W]
        self.lr_masked_list = [self.lr]
        if not (self.W is None) and self.var_delays:
            self.compute_masked_W()
            self.compute_masked_lr()

    def reset_network(self):
        """
        Resets network activity to initial state.
        :return: None
        """
        self.reset_activity()
        self.reset_weights()

    def reset_activity(self , *A):
        if len(A)>0:
            print(self.A)
            self.A = tf.compat.v1.assign(self.A ,A[0])
        else:            

            self.A = tf.Variable(self.A_init ,dtype='float32')


    def reset_weights(self):
        self.W_masked_list = [np.copy(partial_W) for partial_W in self.W_masked_list_init]

    def compute_masked_W(self):
        """
        Creates a list of masked weight matrices, i.e. weight matrices containing only the weights of connections with
        a specified delay.
        Returns: None
        """
        self.W_masked_list.clear()
        for buffStep in range(np.max(self.D) + 1):
            # Create mask for each buffer step
            mask = self.D == buffStep
            # Elementwise product with buffer mask to only add activity to correct buffer step
            buffW = np.multiply(mask, self.W)
            self.W_masked_list.append(buffW)

    def compute_masked_lr(self):
        self.lr_masked_list.clear()
        excitatory_pre = np.repeat(np.expand_dims(np.array(self.n_type > 0, dtype='uint8'), 0), self.N, axis=0)
        for buffStep in range(np.max(self.D) + 1):
            # fix zero weights
            buffLr = self.lr * np.array(self.W_masked_list[buffStep] > 0, dtype='uint8')
            # only update weights with excitatory presynaptic units
            buffLr = buffLr * excitatory_pre
            self.lr_masked_list.append(buffLr)

    def clamp_input(self, input_array):
        """
        Set the input value of the input neurons to that of a given input array.
        :param input_array: ndarray
            N_i by 1 array with N_i the number of input neurons.
        :return: None
        """
        assert input_array.shape.is_compatible_with(self.neurons_in.shape)
        input_ind = self.neurons_in
        input_ind = np.reshape(input_ind, (len(input_ind),))
        # this only works in case of a one dimensional input 
        self.temp_A = self.A
        self.A = tf.compat.v1.assign(self.A[input_ind[0],  0] , input_array)
        self.temp_B = self.A
    
    def build(self, inputs_shape):
        input_size = tf.compat.dimension_value(tf.TensorShape(inputs_shape)[-1])
        if input_size is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]. Shape received is %s"
                % inputs_shape
            )

        def _esn_recurrent_initializer(shape, dtype, partition_info=None):
            recurrent_weights = tf.keras.initializers.get(self.recurrent_initializer)(
                shape, dtype
            )

            connectivity_mask = tf.cast(
                tf.math.less_equal(tf.random.uniform(shape), self.connectivity),
                dtype,
            )
            recurrent_weights = tf.math.multiply(recurrent_weights, connectivity_mask)

            # Satisfy the necessary condition for the echo state property `max(eig(W)) < 1`
            if self.use_norm2:
                # This condition is approximated scaling the norm 2 of the reservoir matrix
                # which is an upper bound of the spectral radius.
                recurrent_norm2 = tf.math.sqrt(
                    tf.math.reduce_sum(tf.math.square(recurrent_weights))
                )
                is_norm2_0 = tf.cast(tf.math.equal(recurrent_norm2, 0), dtype)
                scaling_factor = tf.cast(self.spectral_radius, dtype) / (
                    recurrent_norm2 + 1 * is_norm2_0
                )
            else:
                abs_eig_values = tf.abs(tf.linalg.eig(recurrent_weights)[0])
                scaling_factor = tf.math.divide_no_nan(
                    tf.cast(self.spectral_radius, dtype), tf.reduce_max(abs_eig_values)
                )

            recurrent_weights = tf.multiply(recurrent_weights, scaling_factor)

            return recurrent_weights



        self.rec_initializer = tf.constant_initializer(self.W)
        self.rec_initializer_masked = tf.constant_initializer(np.array(self.W_masked_list))
        self.bias_initializer = tf.constant_initializer(self.WBias)

        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=[self.N, self.N],
            initializer=self.rec_initializer,
            trainable=False,
            dtype=self.dtype,
        )
        self.recurrent_kernel_delayed = self.add_weight(
            name="recurrent_kernel_delayed",
            shape=[self.theta_window, self.N, self.N],
            initializer=self.rec_initializer_masked,
            trainable=False,
            dtype=self.dtype,
        )

        self.bias = self.add_weight(
            name="bias",
            shape=[self.N],
            initializer=self.bias_initializer,
            trainable=False,
            dtype=self.dtype,
        )

        self.A = tf.Variable(self.A_init ,dtype='float32')
        self.built = True

    def call(self, input, state):

        
        # Ensure self.A is a TensorFlow variable
        if len(state.shape)>2:
            if state.shape[2] > 1:
                # Shift from present to past
                state = tf.compat.v1.assign(state[:, :, 1:],state[:, :, :-1])

        else:    
            if state.shape[1] > 1:
                # Shift from present to past
                state = tf.compat.v1.assign(state[:, 1:],state[:, :-1])

        
        if self.B > 1 and self.var_delays:
            in_matrix = state
            
            # Initialize neuron inputs with bias
            neuron_inputs = tf.convert_to_tensor(self.WBias, dtype=tf.float32)  # Add bias weights first

            # Perform delayed recurrent operations
            if len(state.shape)>2:
                delayed_inputs = tf.einsum('dij,tjd->ti', self.recurrent_kernel_delayed, in_matrix)
            else:
                delayed_inputs = tf.einsum('dij,jd->i', self.recurrent_kernel_delayed, in_matrix)

            # Add the delayed inputs to the bias
            neuron_inputs += delayed_inputs
        else:
            in_matrix = tf.cast(state[:, 0], dtype='float32')
            neuron_inputs = tf.matmul(self.recurrent_kernel, in_matrix) + tf.convert_to_tensor(self.bias, dtype=tf.float32)

        # Apply activation function
        y = self.activation_func(neuron_inputs) * tf.constant(self.n_type, dtype='float32')
        
        # Update A with decay
        if len(state.shape)>2:
            A_np = (1 - tf.constant(self.decay, dtype='float32')) * state[:,:, 0] + tf.constant(self.decay, dtype='float32') * y
        else:
            A_np = (1 - tf.constant(self.decay, dtype='float32')) * state[:, 0] + tf.constant(self.decay, dtype='float32') * y
        
        # Assign new values to the first column of A
        if len(state.shape)>2:
            state = tf.compat.v1.assign(state[:,:, 0], A_np)
        else:
            state = tf.compat.v1.assign(state[:, 0], A_np)

        if len(state.shape)>2:
            assert input[0].shape.is_compatible_with(self.neurons_in.shape)
        else:
            assert input.shape.is_compatible_with(self.neurons_in.shape)

        input_ind = self.neurons_in
        input_ind = np.reshape(input_ind, (len(input_ind),))
        
        if len(state.shape)>2:
            state = tf.compat.v1.assign(state[:, -1:, 0], tf.cast(state[:, -1:, 0]*0+input,dtype='float32')) 
        else:
            state = tf.compat.v1.assign(state[-1:, 0], tf.cast(state[0, 0]*0+input[0],dtype='float32')) 

        return state, state

    def update_step(self, input):

        # Ensure self.A is a TensorFlow variable
        if self.A.shape[1] > 1:
            # Shift from present to past
            self.A = tf.compat.v1.assign(self.A[:, 1:],self.A[:, :-1])

        if self.B > 1 and self.var_delays:
            in_matrix = self.A
            
            # Initialize neuron inputs with bias
            neuron_inputs = tf.convert_to_tensor(self.WBias, dtype=tf.float32)  # Add bias weights first

            # Perform delayed recurrent operations
            delayed_inputs = tf.einsum('dij,jd->i', self.recurrent_kernel_delayed, in_matrix)
            self.temp_mul = delayed_inputs

            # Add the delayed inputs to the bias
            neuron_inputs += delayed_inputs
        else:
            in_matrix = tf.cast(self.A[:, 0], dtype='float32')
            neuron_inputs = tf.matmul(self.recurrent_kernel, in_matrix) + tf.convert_to_tensor(self.bias, dtype=tf.float32)

        # Apply activation function
        y = self.activation_func(neuron_inputs) * tf.constant(self.n_type, dtype='float32')
        
        # Update A with decay
        A_np = (1 - tf.constant(self.decay, dtype='float32')) * self.A[:, 0] + tf.constant(self.decay, dtype='float32') * y
        
        # Assign new values to the first column of A
        self.A = tf.compat.v1.assign(self.A[:, 0], A_np)

        assert input.shape.is_compatible_with(self.neurons_in.shape)
        input_ind = self.neurons_in
        input_ind = np.reshape(input_ind, (len(input_ind),))

        self.A = tf.compat.v1.assign(self.A[-1:,  0], tf.cast(self.A[0,  0]*0+input[0],dtype='float32')) 

        
        # Clamp input (assuming clamp_input modifies some state)
        # self.clamp_input(input) 
    
    def get_config(self):
        config = {
            "units": self.units,
            "connectivity": self.connectivity,
            "leaky": self.leaky,
            "spectral_radius": self.spectral_radius,
            "use_norm2": self.use_norm2,
            "use_bias": self.use_bias,
            "activation": tf.keras.activations.serialize(self.activation),
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": tf.keras.initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

def coordinates2distance(coordinates):
    """
    Transforms a spatial configuration of neurons to a distance (adjacency) matrix.
    :param coordinates: ndarray
        N by dims array with N the number of neurons and dims the number of spatial dimensions. Should contain the
        spatial coordinates in a 2D space of each neuron.
    :return: ndarray
        N by N array containing the spatial distance between each neuron.
    """
    N = coordinates.shape[0]

    D = np.zeros((N, N))

    def dist(dist_x, dist_y):
        return np.sqrt(dist_x ** 2 + dist_y ** 2)

    for i in range(N):
        for j in range(N):
            if not i == j:
                dist_x = np.abs(coordinates[i, 0] - coordinates[j, 0])
                dist_y = np.abs(coordinates[i, 1] - coordinates[j, 1])
                d = dist(dist_x, dist_y)
                D[i, j] = d
    return D

def stepwise_activation_tf(neuron_input, threshold=0.0):
    """
    Performs threshold activation on a neuron input array
    :param neuron_input: ndarray
        N by 1 array with N number of neurons that encodes the input of all neurons.
    :param threshold: float
        Threshold value for all neurons
    :return: ndarray
        N by 1 float array with neuron activation.
    """
    x = neuron_input
    y = np.asarray(x > threshold, dtype='float64')
    return y

@tf.function
def sigmoid_activation_tf(neuron_input):
    """
    Performs sigmoid activation on a neuron input array
    :param neuron_input: ndarray
        N by 1 array with N number of neurons that encodes the input of all neurons.
    :return: ndarray
        N by 1 float array with neuron activation.
    """
    x = neuron_input
    y = 1 / (1 + tf.exp(-x))
    return y

def tanh_activation_tf(neuron_input):
    x = neuron_input
    y = tf.tanh(x)
    return y

def elu_tf(neuron_input):
    e = e = tf.constant(2.718281828)
    z = neuron_input
    if z >= 0:
        return z
    else:
        return (e ** z - 1)

def relu_tf(neuron_input):
    return tf.maximum(neuron_input, np.zeros_like(neuron_input))

def sigmoid_der_tf(x):
    return sigmoid_activation_tf(x) * (1-sigmoid_activation_tf(x))

def get_multi_activation_tf(activation_funcs, index_ranges):
    assert len(activation_funcs) == len(index_ranges), 'Number of slices should be equal to number of activation funcs'

    def multi_activation(neuron_input):
        activations = []
        for i, act_f in enumerate(activation_funcs):
            partial_in = neuron_input[index_ranges[i][0]:index_ranges[i][1]]
            if not act_f is None:
                partial_act = act_f(partial_in)
                activations.append(partial_act)
            else:
                activations.append(partial_in)
        activations = np.concatenate(activations)
        return activations

    return multi_activation


class Model:

    def __init__(self, cell , input_data, run_dynamics = False):
 
        # Load the input activity, the target data, and the training mask for this batch of trials

        self.input_data = input_data
        self.cell = cell

        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        self.cell.build(inputs_shape=self.input_data.shape[0])

        if run_dynamics: 
            self.get_network_data_updated(input_data)

        # else:
        #     self.initialize_weights()
        #     self.run_rnn(self.input_data )

        # self.run_dyn_rnn()
        # self.optimize_dyn()        
        
    def initialize_weights(self):
        # Initialize all weights. biases, and initial values
        self.var_dict = {}
        # all keys in par with a suffix of '0' are initial values of trainable variables
        a = ['w_out0','b_out0']
        for k, v in par.items():

            if k in a:
                name = k[:-1]
                self.var_dict[name] = tf.Variable(par[k], name)

    def get_network_data(self, input_data):
        self.cell.build(inputs_shape=input_data.shape[0])

        net_out = []
        network_output_indices = self.cell.neurons_out
        for t in range(input_data.shape[1]):
            inp = np.ones((len(self.cell.neurons_in))) * input_data[:,t]
            self.cell.update_step(inp)
            output = self.cell.A[slice(network_output_indices[0],len(network_output_indices)), 0]
            net_out.append(output)
        self.net_out = tf.stack(net_out, axis=1)
        return net_out

    def get_network_data_updated(self, input_data):

        net_out = []
        net_states = [] 
        states = self.cell.A
        network_output_indices = self.cell.neurons_out

        for t in range(input_data.shape[1]):
            inp = np.ones((len(self.cell.neurons_in))) * input_data[:,t]
            states,_ = self.cell.call(inp, states)
            output = states[slice(network_output_indices[0],len(network_output_indices)), 0]
            net_out.append(output)
            net_states.append(states[slice(network_output_indices[0],len(network_output_indices)), :])

        self.net_out = tf.stack(net_out, axis=1)
        self.net_states = tf.stack(net_states, axis=0)

        return net_out,net_states
        
    def run_rnn(self, input, dtype=tf.float32):
        self.cell = ESNCell(units=par['units'],
                    connectivity=par['connectivity'],
                    leaky=par['leaky'],
                    spectral_radius=par['spectral_radius'],
                    use_norm2=par['use_norm2'],
                    use_bias=par['use_bias'])
        self.cell.build(inputs_shape=input.shape[0])
        states = []
        zero_state = np.zeros((1,par['units']))
        for t in range(input.shape[1]-1):
            if t == 0 :
                h,_ = self.cell.call(inputs=input[:,slice(0,t+1)],state=zero_state)
                states.append(h)
            else:
                h,_ = self.cell.call(inputs=input[:,slice(t,t+1)],state=states[-1])
                states.append(h)
        self.h = states

    def run_dyn_rnn(self):
        # Main model loop

        self.h = []
        self.y = []
        r = np.random.RandomState(400)
        self.cell = ESNCell(units=par['units'],
                            connectivity=par['connectivity'],
                            leaky=par['leaky'],
                            spectral_radius=par['spectral_radius'],
                            use_norm2=par['use_norm2'],
                            use_bias=par['use_bias'])
        # self.cell.build(inputs_shape=1)
        self.h, states = rnn.dynamic_rnn(self.cell, self.input_data, dtype=tf.float32, time_major=True)
        # self.y = tf.tensordot(self.h,self.var_dict['w_out'],axes=1) + self.var_dict['b_out']
        # self.h = tf.stack(self.h)
        # self.y = tf.stack(self.y)

    def warmup(self, input_data):
        for inp in input_data:
            inp = np.ones((len(self.network.neurons_in),)) * inp
            self.network.update_step(inp)

    def optimize_dyn(self):
        self.perf_loss = tf.reduce_mean(tf.squared_difference(self.y, self.target_data))

        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        n = 2 if par['spike_regularization'] == 'L2' else 1
        self.spike_loss = tf.reduce_mean(self.h**n) 
        # self.weight_loss = tf.reduce_mean(tf.nn.relu(self.w_rnn)**n)
        var_list = tf.trainable_variables()
        self.loss = self.perf_loss + par['spike_cost']*self.spike_loss 
                    # + par['weight_cost']*self.weight_loss

        # opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        decay = tf.train.exponential_decay(par['learning_rate'],self.global_step,128,0.9)
        opt = tf.train.MomentumOptimizer(decay,0.9,use_nesterov=False)

        grads_and_vars = opt.compute_gradients(self.loss,var_list)
        capped_gvs = []
        


        for grad, var in grads_and_vars:
          if 'w_out' in var.op.name:
              grad *= par['w_out_mask']

          capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))
        self.train_op = opt.apply_gradients(capped_gvs)

    # def save(self):
    #     """Save the model."""

    #     sess = tf.get_default_session()
    #     path = os.getcwd()+'/model'
    #     os.mkdir(path)
    #     save_path = os.path.join(path, 'model.ckpt')
    #     self.saver.save(sess, save_path)
    #     print("Model saved in file: %s" % save_path)



# if __name__ == "__main__()":

#     INPUT = np.zeros_like(np.random.normal(0, 0.5, size=(1,100)))
#     tf.compat.v1.disable_eager_execution()

#     with  tf.compat.v1.Session() as sess:
#         x = tf.compat.v1.placeholder(tf.float32, [1, 100], name='input_placeholder')

#         esn_model = Model(input_data = x)

#         sess.run(tf.compat.v1.global_variables_initializer())
#         h = sess.run([esn_model.h],feed_dict={x:np.random.normal(0, 0.5, size=(1,100))})
        
#         fps = FixedPointSearch(
#                 ctype = 'ESN',
#                 states = np.expand_dims(np.vstack(h),axis=0),
#                 savepath = '/content', 
#                 cell=esn_model.cell,
#                 sess = sess
#                 )
#         fps.rerun_q_outliers = False
#         fps.sample_states(10,np.vstack(h).reshape([1,99,100]),esn_model.cell,0.9)
#         unique, all_fps = fps.find_fixed_points(inputs = np.zeros([1,1]), save = False)

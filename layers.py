# -*- coding: utf-8 -*-
# #/usr/bin/python2

import tensorflow as tf
import numpy as np
import math


from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops

from functools import reduce
from operator import mul

'''
Some functions are taken directly from Tensor2Tensor Library:
https://github.com/tensorflow/tensor2tensor/
and BiDAF repository:
https://github.com/allenai/bi-att-flow
'''

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

def glu(x):
    """Gated Linear Units from https://arxiv.org/pdf/1612.08083.pdf"""
    x, x_h = tf.split(x, 2, axis = -1)
    return tf.sigmoid(x) * x_h

def noam_norm(x, epsilon=1.0, scope=None, reuse=None):
    """One version of layer normalization."""
    with tf.name_scope(scope, default_name="noam_norm", values=[x]):
        shape = x.get_shape()
        ndims = len(shape)
        return tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(tf.to_float(shape[-1]))

def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias

def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer = regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer = regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

norm_fn = layer_norm#tf.contrib.layers.layer_norm #tf.contrib.layers.layer_norm or noam_norm



def highway(x, size = None, activation = None,
            num_layers = 2, scope = "highway", dropout = 0.0, reuse = None):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name = "input_projection", reuse = reuse)
        for i in range(num_layers):
            T = conv(x, size, bias = True, activation = tf.sigmoid,
                     name = "gate_%d"%i, reuse = reuse)
            H = conv(x, size, bias = True, activation = activation,
                     name = "activation_%d"%i, reuse = reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x


# def feedforward(x):
#     """ We just put the whole implementation of the feedforward neural network used for the prediction of y3 here."""

#     # layer1 = tf.layers.dense(inputs=x, units=3024, activation=tf.nn.relu)
#     # layer2 = tf.layers.dense(inputs=layer1, units=1024, activation=tf.nn.relu)
#     # layer3 = tf.layers.dense(inputs=layer2, units=300, activation=tf.nn.relu)
#     # layer4 = tf.layers.dense(inputs=layer3, units=100, activation=tf.nn.sigmoid)
#     # y3_p = tf.layers.dense(inputs=layer4, units=1, activation=tf.nn.sigmoid)

#     layer1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
#     layer2 = tf.layers.dense(inputs=layer1, units=100, activation=tf.nn.relu)
#     y3_p = tf.layers.dense(inputs=layer2, units=1, activation=tf.nn.sigmoid)
#     y3_p = y3_p[:,0]

#     #y3_p = tf.transpose(y3_p, 0)
#     y3_p = tf.reshape(y3_p, [32])

#     print(y3_p.shape)

#     return y3_p


def layer_dropout(inputs, residual, dropout):
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)


def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, mask = None,
                   num_filters = 128, input_projection = False, num_heads = 8,
                   seq_len = None, scope = "res_block", is_training = True,
                   reuse = None, bias = True, dropout = 0.0):
    """ These "transformer/encoder blocks". 
    Args:
    reuse: defines if this block shares its weights with the previously created residual block.
    
    """
                   
    with tf.variable_scope(scope, reuse = reuse):
        if input_projection:
            inputs = conv(inputs, num_filters, name = "input_projection", reuse = reuse)

        outputs = inputs
        sublayer = 1
        total_sublayers = (num_conv_layers + 2) * num_blocks

        # create the number of residual blocks requested: each composed of multiple convolution blocks and and attention block
        for i in range(num_blocks):
            # position encoding:
            outputs = add_timing_signal_1d(outputs)

            # the convolution layers
            outputs, sublayer = conv_block(outputs, num_conv_layers, kernel_size, num_filters,
                seq_len = seq_len, scope = "encoder_block_%d"%i,reuse = reuse, bias = bias,
                dropout = dropout, sublayers = (sublayer, total_sublayers))

            # the attention block
            outputs, sublayer = self_attention_block(outputs, num_filters, seq_len, mask = mask, num_heads = num_heads,
                scope = "self_attention_layers%d"%i, reuse = reuse, is_training = is_training,
                bias = bias, dropout = dropout, sublayers = (sublayer, total_sublayers))

        return outputs


def conv_block(inputs, num_conv_layers, kernel_size, num_filters,
               seq_len = None, scope = "conv_block", is_training = True,
               reuse = None, bias = True, dropout = 0.0, sublayers = (1, 1)):
    """ The convolution block of the residual block. """

    with tf.variable_scope(scope, reuse = reuse):
        outputs = tf.expand_dims(inputs,2)
        l, L = sublayers

        # we create as many convolutional layers as requested:
        for i in range(num_conv_layers):
            residual = outputs

            # we perform the layernorm:
            outputs = norm_fn(outputs, scope = "layer_norm_%d"%i, reuse = reuse)

            # we add the dropout on every second convlayer:
            if (i) % 2 == 0:
                outputs = tf.nn.dropout(outputs, 1.0 - dropout)

            # we create the output of that layer by performing a depthwise_separable_convolution
            outputs = depthwise_separable_convolution(outputs, kernel_size = (kernel_size, 1), num_filters = num_filters,
                scope = "depthwise_conv_layers_%d"%i, is_training = is_training, reuse = reuse)

            outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
            l += 1

        return tf.squeeze(outputs,2), l


def self_attention_block(inputs, num_filters, seq_len, mask = None, num_heads = 8,
                         scope = "self_attention_ffn", reuse = None, is_training = True,
                         bias = True, dropout = 0.0, sublayers = (1, 1)):
    """ The attention block of the residual block.
        Composed of a self-attention layer and a feed forward network.

    """

    with tf.variable_scope(scope, reuse = reuse):
        l, L = sublayers

        # Self attention part
        outputs = norm_fn(inputs, scope = "layer_norm_1", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)

        outputs = multihead_attention(outputs, num_filters,
            num_heads = num_heads, seq_len = seq_len, reuse = reuse,
            mask = mask, is_training = is_training, bias = bias, dropout = dropout)

        residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
        l += 1

        # Feed-forward part
        outputs = norm_fn(residual, scope = "layer_norm_2", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name = "FFN_1", reuse = reuse)
        outputs = conv(outputs, num_filters, True, None, name = "FFN_2", reuse = reuse)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        l += 1

        return outputs, l


def multihead_attention(queries, units, num_heads, memory = None, seq_len = None,
                        scope = "Multi_Head_Attention", reuse = None, mask = None,
                        is_training = True, bias = True, dropout = 0.0):
    """ Multi-head attention as described in "Attention is all you need".
    
    Args:

    """
    with tf.variable_scope(scope, reuse = reuse):
        # Self attention
        if memory is None:
            memory = queries
        
        # print("memory/queries: {}".format(memory.get_shape()))
        # print("memory/queries: {}".format(queries.get_shape()))

        memory = conv(memory, 2 * units, name = "memory_projection", reuse = reuse)
        # print("memory: {}".format(memory.get_shape()))
        query = conv(queries, units, name = "query_projection", reuse = reuse)
        # print("query: {}".format(query.get_shape()))
        Q = split_last_dimension(query, num_heads)
        # print("Q : {}".format(Q.get_shape()))
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory,2,axis = 2)]
        # print("K : {}".format(K.get_shape()))
        # print("V : {}".format(V.get_shape()))

        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head**-0.5

        x = dot_product_attention(Q,K,V, bias = bias, seq_len = seq_len, mask = mask,
                                  is_training = is_training, scope = "dot_product_attention",
                                  reuse = reuse, dropout = dropout)
        
        # print("x : {}".format(x.get_shape()))

        return combine_last_two_dimensions(tf.transpose(x,[0,2,1,3]))


def dot_product_attention(q, k, v, bias,
                          seq_len = None,
                          mask = None,
                          is_training = True,
                          scope=None,
                          reuse = None,
                          dropout = 0.0):
    """ Dot-product attention as described in "Attention is all you need".
    Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    is_training: a bool of training
    scope: an optional string

    Returns:
    The attention tensor.
    """
    with tf.variable_scope(scope, default_name="dot_product_attention", reuse = reuse):
        # [batch, num_heads, query_length, memory_length]
        # We compute the logit weights: Note that q was already scaled
        #  by the square root of the dimention in multihead_attention()
        logits = tf.matmul(q, k, transpose_b=True)
        # if required, we add a bias:
        if bias:
            b = tf.get_variable("bias", logits.shape[-1],
                    regularizer=regularizer, initializer = tf.zeros_initializer())
            logits += b
        # if there is a mask we apply it to the loggits obtained:
        if mask is not None:
            shapes = [x  if x != None else -1 for x in logits.shape.as_list()]
            mask = tf.reshape(mask, [shapes[0],1,1,shapes[-1]])
            logits = mask_logits(logits, mask)
        # computing the softmax of the logits to obtain the weights:
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout)
        
        return tf.matmul(weights, v)


def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    """ The convolution function used throughout the model.

        Important to notice that it is used as a feedforward network with variable input size in some places.
    
    """
    with tf.variable_scope(name, reuse = reuse):

        shapes = inputs.shape.as_list()

        # we define the filter, bias and stride shapes in function of the number of input dimentions:
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1

        # define the convolution function depending on the size:
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        # we define the kernel and perform the convolution using it:
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")

        # if we use a bias: add it on the output
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())

        if activation is not None:
            return activation(outputs)
        else:
            return outputs




def conv_2(inputs, strides, output_size = 1, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    """ The convolution function used throughout the model.

    Important to notice that it is used as a feedforward network with variable input size in some places.
    
    """
    with tf.variable_scope(name, reuse = reuse):
        this_initializer = tf.glorot_uniform_initializer()

        shapes = inputs.shape.as_list()

        # we define the filter, bias and stride shapes in function of the number of input dimentions:
        filter_shape = [kernel_size,shapes[-1],shapes[-1]]
        bias_shape = [1,1,output_size]
        strides = strides #3
        #print("ho!!!!")

        # define the convolution function:
        conv_func = tf.nn.conv1d

        # we define the kernel and perform the convolution using it:
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = this_initializer)
        outputs = conv_func(inputs, kernel_, strides, "VALID")

        # if we use a bias: add it on the output
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())

        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def mask_logits(inputs, mask, mask_value = -1e30):
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope = "depthwise_separable_convolution",
                                    bias = True, is_training = True, reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable("depthwise_filter",
                                        (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = initializer_relu())
        pointwise_filter = tf.get_variable("pointwise_filter",
                                        (1,1,shapes[-1],num_filters),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = initializer_relu())
        outputs = tf.nn.separable_conv2d(inputs,
                                        depthwise_filter,
                                        pointwise_filter,
                                        strides = (1,1,1,1),
                                        padding = "SAME")
        if bias:
            b = tf.get_variable("bias",
                    outputs.shape[-1],
                    regularizer=regularizer,
                    initializer = tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret,[0,2,1,3])


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
    x: a Tensor with shape [..., a, b]
    Returns:
    a Tensor with shape [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal

def ndim(x):
    """Copied from keras==2.0.6
    Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(inputs)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None

def dot(x, y):
    """Modified from keras==2.0.6
    Multiplies 2 tensors (and/or variables) and returns a *tensor*.

    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out

def batch_dot(x, y, axes=None):
    """Copy from keras==2.0.6
    Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out

def optimized_trilinear_for_attention(args, c_maxlen, q_maxlen, input_keep_prob=1.0,
    scope='efficient_trilinear',
    bias_initializer=tf.zeros_initializer(),
    kernel_initializer=initializer()):
    assert len(args) == 2, "just use for computing attention with two input"
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("`args` must be 3 dims (batch_size, len, dimension)")
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("the last dimension of `args` must equal")
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    droped_args = [tf.nn.dropout(arg, input_keep_prob) for arg in args]
    with tf.variable_scope(scope):
        weights4arg0 = tf.get_variable(
            "linear_kernel4arg0", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4arg1 = tf.get_variable(
            "linear_kernel4arg1", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4mlu = tf.get_variable(
            "linear_kernel4mul", [1, 1, arg_size],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        biases = tf.get_variable(
            "linear_bias", [1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=bias_initializer)
        subres0 = tf.tile(dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])
        subres1 = tf.tile(tf.transpose(dot(droped_args[1], weights4arg1), perm=(0, 2, 1)), [1, c_maxlen, 1])
        subres2 = batch_dot(droped_args[0] * weights4mlu, tf.transpose(droped_args[1], perm=(0, 2, 1)))
        res = subres0 + subres1 + subres2
        nn_ops.bias_add(res, biases)
        return res

def trilinear(args,
            output_size = 1,
            bias = True,
            squeeze=False,
            wd=0.0,
            input_keep_prob= 1.0,
            scope = "trilinear"):
    with tf.variable_scope(scope):
        flat_args = [flatten(arg, 1) for arg in args]
        flat_args = [tf.nn.dropout(arg, input_keep_prob) for arg in flat_args]
        flat_out = _linear(flat_args, output_size, bias, scope=scope)
        out = reconstruct(flat_out, args[0], 1)
        return tf.squeeze(out, -1)

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat

def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

def _linear(args,
            output_size,
            bias,
            bias_initializer=tf.zeros_initializer(),
            scope = None,
            kernel_initializer=initializer(),
            reuse = None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]
  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope, reuse = reuse) as outer_scope:
    weights = tf.get_variable(
        "linear_kernel", [total_arg_size, output_size],
        dtype=dtype,
        regularizer=regularizer,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable(
          "linear_bias", [output_size],
          dtype=dtype,
          regularizer=regularizer,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)


def total_params():
    """ Compute and prints the total number of parameters in the model. """
    total_parameters = 0
    for variable in tf.trainable_variables():
        #for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Input_Embedding_Layer'):
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))
    
    # print the number of parameters of each scope:
    list_of_scopes = ['Input_Embedding_Layer','Embedding_Encoder_Layer','Context_to_Query_Attention_Layer','Answerability_Prediction','Model_Encoder_Layer','Output_Layer']

    for scope in list_of_scopes:

        total_parameters = 0
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total number of trainable parameters " + scope + ": {}".format(total_parameters))



import tensorflow as tf


def weight_variable(name, shape):
    """Create a weight variable with appropriate initialization."""
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable(name + 'W', dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable(name + 'b', dtype=tf.float32,
                           initializer=initial)


def batch_norm(x, name, phase=None):
    """Create a Batch Norm layer"""
    return tf.contrib.layers.batch_norm(x,
                                        center=True, scale=True,
                                        is_training=phase,
                                        scope=name + 'bn')


def fc_layer(bottom, out_dim, name, add_reg=False, use_relu=True):
    """Create a fully connected layer"""
    in_dim = bottom.get_shape()[1]
    with tf.variable_scope(name):
        weights = weight_variable(name, shape=[in_dim, out_dim])
        tf.summary.histogram('histogram', weights)
        biases = bias_variable(name, [out_dim])
        layer = tf.matmul(bottom, weights)
        layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def conv_2d(inputs, filter_size, stride, num_inChannel, num_filters, name, add_reg=False, use_relu=True):
    """Create a convolution layer."""
    with tf.variable_scope(name):
        shape = [filter_size, filter_size, num_inChannel, num_filters]
        weights = weight_variable(name, shape=shape)
        tf.summary.histogram('histogram', weights)
        biases = bias_variable(name, [num_filters])
        layer = tf.nn.conv2d(input=inputs,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        print('{}: {}'.format(layer.name, layer.get_shape()))
        layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def flatten_layer(layer):
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def max_pool(x, ksize, stride, name):
    """Create a max pooling layer."""
    maxpool = tf.nn.max_pool(x,
                             ksize=[1, ksize, ksize, 1],
                             strides=[1, stride, stride, 1],
                             padding="SAME",
                             name=name)
    print('{}: {}'.format(maxpool.name, maxpool.get_shape()))
    return maxpool


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

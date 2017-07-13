import numpy as np
import random
import scipy
import tensorflow as tf


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def reformat(x, y, img_size, num_ch, num_class):
    """ Reformats the data to the format acceptable for 2D conv layers"""
    dataset = x.reshape(
        (-1, img_size, img_size, num_ch)).astype(np.float32)
    labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
    return dataset, labels


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def random_rotation_2d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).
    Arguments:
    max_angle: `float`. The maximum rotation angle.
    Returns:
    batch of rotated 2D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image = np.squeeze(batch[i])
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)


def accuracy_generator(labels_tensor, logits_tensor):
    """
     Calculates the classification accuracy.
     Note that this is for multi-task classification.
    :param labels_tensor: Tensor of correct predictions of size [batch_size, numConditions*2]
    :param logits_tensor: Predicted scores (logits) by the model.
            It should have the same dimensions as labels_tensor
    :return: accuracy: tensor of size numConditions which gives the accuracy for each condition
             avg_accuracy: average accuracy across all conditions
    """

    labels_series = tensor_to_series(labels_tensor)
    logits_series = tensor_to_series(logits_tensor)
    correct_pred = [tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
                    for logits, labels in zip(logits_series, labels_series)]
    accuracy_list = [tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                     for correct_prediction in correct_pred]
    accuracy = tf.stack(accuracy_list, axis=0)
    avg_accuracy = tf.reduce_mean(accuracy)
    return accuracy, avg_accuracy


def cross_entropy_loss(labels_tensor, logits_tensor):
    """
     Calculates the cross-entropy loss function for the given parameters.
     Note that this is for multi-task classification problem.
    :param labels_tensor: Tensor of correct predictions of size [batch_size, numConditions*2]
    :param logits_tensor: Predicted scores (logits) by the model.
            It should have the same dimensions as labels_tensor
    :return: Cross-entropy Loss tensor
    """

    labels_series = tensor_to_series(labels_tensor)
    logits_series = tensor_to_series(logits_tensor)
    losses_list = [tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                   for logits, labels in zip(logits_series, labels_series)]
    losses = tf.stack(losses_list, axis=1)
    total_loss = tf.reduce_mean(losses)
    return total_loss


def tensor_to_series(input_tensor):
    """
     Converts the input tensor to a list of tensors by unstacking it.
    :param input_tensor: Input tensor of size [batch_size, numConditions*2]
    :return: List of tensors of length numConditions where each component of the list
    is of size: [batch_size, 2]
    """
    [batch_size, _] = input_tensor.get_shape().as_list()
    tensor_reshaped = tf.reshape(input_tensor, [batch_size, -1, 2])
    tensor_transp = tf.transpose(tensor_reshaped, perm=[0, 2, 1])
    input_series = tf.unstack(tensor_transp, axis=2)
    return input_series


def logits_to_probs(input_tensor):
    """
     Converts the tensor of logits to probabilities by passing it through
     the softmax function.
    :param input_tensor: Logits tensor of size [batch_size, numConditions*2]
    :return: Tensor of probability values with size similar to input tensor
    """
    [batch_size, numCond2] = input_tensor.get_shape().as_list()
    numConditions = numCond2 / 2
    logits_series = tensor_to_series(input_tensor)
    probs_list = [tf.nn.softmax(logits) for logits in logits_series]
    _stacked = tf.stack(probs_list, axis=2)  # Tensor of shape [batch_size, 2, numConditions]
    probs = _stacked[:, :, 0]
    for i in range(1, numConditions):
        probs = tf.concat([probs, _stacked[:, :, i]], axis=1)
    return probs

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:55:42 2017

@author: Aryan Mobiny
"""

import tensorflow as tf
from ops import *
from utils import *


class Alexnet:

    # Class properties
    __network = None         # Graph for AlexNet
    __train_op = None        # Operation used to optimize loss function
    __loss = None            # Loss function to be optimized, which is based on predictions
    __accuracy = None        # Classification accuracy for all conditions
    __avg_accuracy = None    # Average classification accuracy over all conditions
    __probs = None           # Prediction probability matrix of shape [batch_size, numConditions*2]

    def __init__(self, numConditions, imgSize, imgChannel,batchSize):
        self.imgSize = imgSize
        self.numConditions = numConditions
        self.imgChannel = imgChannel
        self.h1 = 500
        self.h2 = 200
        self.init_lr = 0.001
        self.batch_size = batchSize
        self.x, self.y, self.keep_prob = self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size,
                                                       self.imgSize,
                                                       self.imgSize,
                                                       self.imgChannel), name='x-input')
            self.y = tf.placeholder(tf.float32, shape=(self.batch_size
                                                       , self.numConditions*2), name='y-input')
            self.keep_prob = tf.placeholder(tf.float32)
        return self.x, self.y, self.keep_prob

    def inference(self):
        if self.__network:
            return self
        # Building network...
        net = conv_2d(self.x, 11, 4, 1, 96, 'CONV1', add_reg=False, use_relu=True)
        net = max_pool(net, 3, 2, 'MaxPool1')
        net = lrn(net, 2, 2e-05, 0.75, name='norm1')
        net = conv_2d(net, 5, 1, 96, 256, 'CONV2', add_reg=False, use_relu=True)
        net = max_pool(net, 3, 2, 'MaxPool2')
        net = lrn(net, 2, 2e-05, 0.75, name='norm2')
        net = conv_2d(net, 3, 1, 256, 384, 'CONV3', add_reg=False, use_relu=True)
        net = conv_2d(net, 3, 1, 384, 384, 'CONV4', add_reg=False, use_relu=True)
        net = conv_2d(net, 3, 1, 384, 256, 'CONV5', add_reg=False, use_relu=True)
        net = max_pool(net, 3, 2, 'MaxPool3')
        layer_flat = flatten_layer(net)
        net = fc_layer(layer_flat, self.h1, 'FC1', add_reg=False, use_relu=True)
        net = dropout(net, self.keep_prob)
        net = fc_layer(net, self.h2, 'FC2', add_reg=False, use_relu=True)
        net = dropout(net, self.keep_prob)
        net = fc_layer(net, self.numConditions*2, 'FC3', add_reg=False, use_relu=False)
        self.__network = net
        return self

    def pred_func(self):
        if self.__probs:
            return self
        self.__probs = logits_to_probs(self.__network)
        return self

    def accuracy_func(self):
        if self.__accuracy:
            return self
        with tf.name_scope('Accuracy'):
            self.__accuracy, self.__avg_accuracy = accuracy_generator(self.y, self.__network)
            tf.summary.scalar('accuracy', self.__avg_accuracy)
        return self

    def loss_func(self):
        if self.__loss:
            return self
        with tf.name_scope('Loss'):
            self.__loss = cross_entropy_loss(self.y, self.__network)
            tf.summary.scalar('cross_entropy', self.__loss)
        return self

    def train_func(self):
        if self.__train_op:
            return self
        with tf.name_scope('Train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr)
            self.__train_op = optimizer.minimize(self.__loss)
        return self

    @property
    def probs(self):
        return self.__probs

    @property
    def network(self):
        return self.__network

    @property
    def train_op(self):
        return self.__train_op

    @property
    def loss(self):
        return self.__loss

    @property
    def accuracy(self):
        return self.__accuracy

    @property
    def avg_accuracy(self):
        return self.__avg_accuracy
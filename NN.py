# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:18:04 2020

@author: Ashley
"""

import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np

def net(x, name, _extra_train_ops, n_neuron, zero=False, train = True):
    with tf.variable_scope(name):
        x_norm, _extra_train_ops = _batch_norm(
            x, _extra_train_ops, name="layer0_normalization", train = train
        )
        hidden = x_norm
        for i in range(1, len(n_neuron)):
            hidden, _extra_train_ops = _one_layer(
                hidden,
                _extra_train_ops,
                n_neuron[i],
                name="layer" + str(i),
                zero=zero,
                train=train
            )
        z, _extra_train_ops = _one_layer(
            hidden,
            _extra_train_ops,
            n_neuron[-1],
            activation_fn=None,
            name="final",
            zero=zero,
            train=train
        )
    if train:
        return z, _extra_train_ops
    else:
        return z


def _one_layer(
    input_,
    _extra_train_ops,
    output_size,
    activation_fn=tf.nn.tanh,
    stddev=5.0,
    name="linear",
    zero=False,
    train = True
):
    with tf.variable_scope(name):
        shape = input_.get_shape().as_list()
        initializer = (
            tf.zeros_initializer()
            if zero
            else tf.random_normal_initializer(
                stddev=stddev / np.sqrt(shape[1] + output_size)
            )
        )
        w = tf.get_variable("Matrix", [shape[1], output_size], tf.float64, initializer)
        # e = tf.get_variable("Bias", [output_size], tf.float64, initializer)
        hidden = input_ @ w
        hidden_bn, _extra_train_ops = _batch_norm(
            hidden, _extra_train_ops, name="normalization", train = train
        )
        if activation_fn:
            return activation_fn(hidden_bn), _extra_train_ops
        else:
            return hidden_bn, _extra_train_ops


def _batch_norm(x, _extra_train_ops, name, train):
    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]
        beta = tf.get_variable(
            "beta",
            params_shape,
            tf.float64,
            initializer=tf.random_normal_initializer(0.0, stddev=0.1),
        )
        gamma = tf.get_variable(
            "gamma",
            params_shape,
            tf.float64,
            initializer=tf.random_uniform_initializer(0.1, 0.5),
        )
        moving_mean = tf.get_variable(
            "moving_mean",
            params_shape,
            tf.float64,
            initializer=tf.constant_initializer(0.0),
            trainable=False,
        )
        moving_variance = tf.get_variable(
            "moving_variance",
            params_shape,
            tf.float64,
            initializer=tf.constant_initializer(1.0),
            trainable=False,
        )
        mean, variance = tf.nn.moments(x, [0], name="moments")
        if train:
            _extra_train_ops.append(
                moving_averages.assign_moving_average(moving_mean, mean, 0.99)
            )
            _extra_train_ops.append(
                moving_averages.assign_moving_average(moving_variance, variance, 0.99)
            )
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-6)
        y.set_shape(x.get_shape())
        return y, _extra_train_ops





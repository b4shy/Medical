import tensorflow as tf
import numpy as np


class SegNetBasic:
    INPUT_SHAPE = [320, 320, 80]

    def __init__(self, no_classes):
        self.classes = no_classes

    def predict(self, data, is_training):
        d1 = self._decoder_block(data, 16, is_training)
        print(d1)
        d2 = self._decoder_block(d1, 16, is_training)
        print(d2)
        d3 = self._decoder_block(d2, 32, is_training)
        print(d3)
        d4 = self._decoder_block(d3, 32, is_training)
        print(d4)

        e1 = self._encoder_block(d4, 32, is_training)
        e2 = self._encoder_block(e1, 16, is_training)
        e3 = self._encoder_block(e2, 16, is_training)
        e4 = self._encoder_block(e3, self.classes, is_training)

        return e4

    def loss(self, image, mask):
        class_weight = tf.constant([0.01, 1])
        logits = tf.multiply(image, class_weight)
        print(logits)
        recon_error = tf.nn.softmax_cross_entropy_with_logits(labels=mask, logits=image)
        print(recon_error)

        cost = tf.reduce_mean(recon_error)
        return cost

    def optimizer(self, lr=1e-5):
        return tf.train.AdamOptimizer(learning_rate=lr)

    def _decoder_block(self, data, no_filters, is_training):
        l1 = tf.layers.conv3d(data, no_filters, [3, 3, 3], padding="same")
        m1 = tf.layers.max_pooling3d(l1, [2, 2, 2], [2, 2, 2], padding="same")
        o1 = tf.nn.relu(m1)
        return o1

    def _encoder_block(self, data, no_filters, is_training):
        u1 = tf.layers.conv3d_transpose(data, no_filters, kernel_size=[7, 7, 7], strides=[2, 2, 2], padding="same")

        return u1

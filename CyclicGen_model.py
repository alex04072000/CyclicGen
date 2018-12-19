"""Implements a voxel flow model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.loss_utils import l1_loss, l2_loss, vae_loss
from utils.geo_layer_utils import vae_gaussian_layer
from utils.geo_layer_utils import bilinear_interp
from utils.geo_layer_utils import meshgrid

FLAGS = tf.app.flags.FLAGS
epsilon = 0.001


class Voxel_flow_model(object):
    def __init__(self, is_train=True, is_extrapolation=False):
        self.is_train = is_train
        self.is_extrapolation = is_extrapolation

    def inference(self, input_images):
        """Inference on a set of input_images.
        Args:
        """
        return self._build_model(input_images)

    def total_var(self, images):
        pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
        pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
        tot_var = (tf.reduce_mean(tf.sqrt(tf.square(pixel_dif1) + epsilon**2)) + tf.reduce_mean(tf.sqrt(tf.square(pixel_dif2) + epsilon**2)))
        return tot_var

    def loss(self, predictions, targets):
        """Compute the necessary loss for training.
        Args:
        Returns:
        """
        # self.reproduction_loss = l1_loss(predictions, targets)
        self.reproduction_loss = tf.reduce_mean(tf.sqrt(tf.square(predictions - targets) + epsilon**2))

        self.motion_loss = self.total_var(self.flow)
        self.mask_loss = self.total_var(self.mask)

        # return [self.reproduction_loss, self.prior_loss]
        return self.reproduction_loss + 0.01 * self.motion_loss + 0.005 * self.mask_loss

    def l1loss(self, predictions, targets):
        self.reproduction_loss = l1_loss(predictions, targets)
        return self.reproduction_loss

    def _build_model(self, input_images):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            # Define network
            batch_norm_params = {
                'decay': 0.9997,
                'epsilon': 0.001,
                'is_training': self.is_train,
            }
            with slim.arg_scope([slim.batch_norm], is_training=self.is_train, updates_collections=None):
                with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params):
                    x0 = slim.conv2d(input_images, 64, [5, 5], stride=1, scope='conv1')
                    # with tf.name_scope('conv1') as scope:
                    #     kernel = tf.Variable(tf.truncated_normal([5, 5, 8, 64], dtype=tf.float32, stddev=1e-1), name='weights')
                    #     conv = tf.nn.atrous_conv2d(input_images, kernel, 2, padding='SAME')
                    #     x0 = tf.nn.relu(conv, name=scope)

                    net = slim.max_pool2d(x0, [2, 2], scope='pool1')
                    x1 = slim.conv2d(net, 128, [5, 5], stride=1, scope='conv2')

                    net = slim.max_pool2d(x1, [2, 2], scope='pool2')
                    x2 = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3')

                    net = slim.max_pool2d(x2, [2, 2], scope='pool3')
                    net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')

                    net = tf.image.resize_bilinear(net, [x2.get_shape().as_list()[1], x2.get_shape().as_list()[2]])
                    net = slim.conv2d(tf.concat([net, x2], -1), 256, [3, 3], stride=1, scope='conv5')

                    net = tf.image.resize_bilinear(net, [x1.get_shape().as_list()[1], x1.get_shape().as_list()[2]])
                    net = slim.conv2d(tf.concat([net, x1], -1), 128, [3, 3], stride=1, scope='conv6')

                    net = tf.image.resize_bilinear(net, [x0.get_shape().as_list()[1], x0.get_shape().as_list()[2]])
                    y0 = slim.conv2d(tf.concat([net, x0], -1), 64, [5, 5], stride=1, scope='conv7')


        net = slim.conv2d(y0, 3, [5, 5], stride=1, activation_fn=tf.tanh,
                          normalizer_fn=None, scope='conv8')
        net_copy = net

        flow = net[:, :, :, 0:2]
        mask = tf.expand_dims(net[:, :, :, 2], 3)

        self.flow = flow


        grid_x, grid_y = meshgrid(x0.get_shape().as_list()[1], x0.get_shape().as_list()[2])
        grid_x = tf.tile(grid_x, [FLAGS.batch_size, 1, 1])
        grid_y = tf.tile(grid_y, [FLAGS.batch_size, 1, 1])

        flow = 0.5 * flow

        flow_ratio = tf.constant([255.0 / (x0.get_shape().as_list()[2]-1), 255.0 / (x0.get_shape().as_list()[1]-1)])
        flow = flow * tf.expand_dims(tf.expand_dims(tf.expand_dims(flow_ratio, 0), 0), 0)

        if self.is_extrapolation:
            coor_x_1 = grid_x + flow[:, :, :, 0] * 2
            coor_y_1 = grid_y + flow[:, :, :, 1] * 2
            coor_x_2 = grid_x + flow[:, :, :, 0]
            coor_y_2 = grid_y + flow[:, :, :, 1]
        else:
            coor_x_1 = grid_x + flow[:, :, :, 0]
            coor_y_1 = grid_y + flow[:, :, :, 1]
            coor_x_2 = grid_x - flow[:, :, :, 0]
            coor_y_2 = grid_y - flow[:, :, :, 1]

        output_1 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1, coor_y_1, 'interpolate')
        output_2 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2, coor_y_2, 'interpolate')

        self.warped_img1 = output_1
        self.warped_img2 = output_2

        self.warped_flow1 = bilinear_interp(-flow[:, :, :, 0:3]*0.5, coor_x_1, coor_y_1, 'interpolate')
        self.warped_flow2 = bilinear_interp(flow[:, :, :, 0:3]*0.5, coor_x_2, coor_y_2, 'interpolate')

        mask = 0.5 * (1.0 + mask)
        self.mask = mask
        mask = tf.tile(mask, [1, 1, 1, 3])
        net = tf.multiply(mask, output_1) + tf.multiply(1.0 - mask, output_2)

        return [net, net_copy]

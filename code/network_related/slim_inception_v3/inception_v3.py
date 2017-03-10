import os
import numpy as np
import random
import time
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_arg_scope

class inception_v3(object):

    def __init__(self):
        self.timestamp = time.strftime('%Y%m%d-%H%M', time.localtime())



    def build_graph(self, own_layers=1):
        g = tf.get_default_graph()

        # Define preprocessing
        with tf.name_scope('Preprocessing'):
            input_im = tf.placeholder(tf.uint8, shape=[None, None, None, 3], name='input_im')
            resized_im = tf.image.resize_bilinear(tf.image.convert_image_dtype(
                tf.convert_to_tensor(input_im), dtype=tf.float32), [299, 299], name='resized_im')
            normalized_im = tf.mul(tf.sub(resized_im, 0.5), 2.0, name='normalized_im')

        # Load Inception-v3 graph from slim
        with slim.arg_scope(inception_v3_arg_scope()):
            logits, end_points = inception_v3(normalized_im, num_classes=1001, is_training=False)
        slim_variables = [v for v in tf.global_variables() if v.name.startswith('InceptionV3/')]

        # Create own branch




    def train(self, batch_size=200)
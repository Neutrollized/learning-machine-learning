#!/usr/bin/env python3

#######################################################
#
# Cats & Dogs with (Color) CNNs
# using Transfer Learning
#
#######################################################

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image	# Python Image Library

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


#---------------------------------------
# download mobilenet classifier
#---------------------------------------
CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_RES = 224

model = tf.keras.Sequential([
  hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
])


#--------------------------------------
# import cats & dogs datasets
#--------------------------------------

# NOTE: 'train[:80%]' = from start up to 80%
#       'train[80%:]' = from 80% up to the end (i.e. last 20%)
# https://www.tensorflow.org/datasets/splits
(train_examples, validation_examples), info = tfds.load('cats_vs_dogs',
                                                        with_info=True,
                                                        as_supervised=True,
                                                        split=['train[:80%]', 'train[80%:]']
                                                       )

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

# you can see that the images are not all the same size
for i, example_image in enumerate(train_examples.take(3)):
  print("Image {} shape: {}".format(i+1, example_image[0].shape))

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches      = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

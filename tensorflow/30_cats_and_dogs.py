#!/usr/bin/env python3

#######################################################
#
# Cats & Dogs with (Color) CNNs
#
#######################################################

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# helper libraries
import os
import matplotlib.pyplot as plt
import numpy as np

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#--------------------------------------
# import cats & dogs datasets
#--------------------------------------

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=True)


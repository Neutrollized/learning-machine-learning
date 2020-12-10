#!/usr/bin/env python3

#######################################################
#
# Transfer Learning (military uniforms)
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

# download the ImageNet labels
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())


#---------------------------------------------------
# using mobilenet for prediction (single image)
#---------------------------------------------------

grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))
print(grace_hopper)

grace_hopper = np.array(grace_hopper)/255.0
print(grace_hopper.shape)

result = model.predict(grace_hopper[np.newaxis, ...])
print(result.shape)

predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
plt.show()

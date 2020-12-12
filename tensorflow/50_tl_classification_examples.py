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

CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
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
# you'll see that in this case that there are 1001 logits (or function that maps probabilities) that map
# to a rating of each probably class of image (i.e. cat, dog, microwave, etc.)
print(result.shape)

# https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
# we're going to pull the highest probably value
predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
plt.show()


#-----------------------------------------------------------------------
# classifying the different breeds of cats & dogs (multiple images) 
#-----------------------------------------------------------------------
# NOTE: 'train[:80%]' = from start up to 80%
#       'train[80%:]' = from 80% up to the end (i.e. last 20%)
# https://www.tensorflow.org/datasets/splits
(train_examples, validation_examples), info = tfds.load('cats_vs_dogs',
                                                        with_info=True,
                                                        as_supervised=True,
                                                        split=['train[:80%]', 'train[80%:]']
                                                       )

num_examples = info.splits['train'].num_examples
num_classes  = info.features['label'].num_classes

# you can see that the images are not all the same size
for i, example_image in enumerate(train_examples.take(3)):
  print("Image {} shape: {}".format(i+1, example_image[0].shape))

# and hence the need to resize them all
def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches      = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

result_batch = model.predict(image_batch)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
print(predicted_class_names)

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")
plt.show()

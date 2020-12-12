#!/usr/bin/env python3

##################################################
#
# Loading Saved Models
#
##################################################

import tensorflow as tf
from tensorflow import keras

import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


#-------------------------------------------------------------------
# download mobilenet classifier
# - this part is common between both the cats_and_dogs
#   and flower_classification model as they both used MobileNet
#-------------------------------------------------------------------

CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
IMAGE_RES = 224

model = tf.keras.Sequential([
  hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
])

# download the ImageNet labels
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())


#---------------------------
# cats & dogs
# load saved model
#---------------------------

# https://github.com/tensorflow/tensorflow/issues/26835
catdog_sm = keras.models.load_model("./saved_models/cats_and_dogs.h5",
                                    custom_objects={'KerasLayer':hub.KerasLayer}
                                   )

print(catdog_sm.summary())


#--------------------------------------
# import cats & dogs datasets
#--------------------------------------

(train_examples, validation_examples), info = tfds.load('cats_vs_dogs',
                                                        with_info=True,
                                                        as_supervised=True,
                                                        split=['train[:80%]', 'train[80%:]']
                                                       )

num_examples = info.splits['train'].num_examples
num_classes  = info.features['label'].num_classes

#for i, example_image in enumerate(train_examples.take(3)):
#  print("Image {} shape: {}".format(i+1, example_image[0].shape))

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches      = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)


#-----------------------------------------
# check loaded model's predictions
#-----------------------------------------

class_names = np.array(info.features['label'].names)
print(class_names)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch       = catdog_sm.predict(image_batch)
predicted_batch       = tf.squeeze(predicted_batch).numpy()
predicted_ids         = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]
print(predicted_class_names)

print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  plt.title(predicted_class_names[n].title(), color=color)
  plt.axis('off')

_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
plt.show()


###########################################################################################

#---------------------------
# flowers 
# load saved model
#---------------------------

# https://github.com/tensorflow/tensorflow/issues/26835
flowers_sm = keras.models.load_model("./saved_models/flower_classification.h5",
                                     custom_objects={'KerasLayer':hub.KerasLayer}
                                    )

print(flowers_sm.summary())


#--------------------------------------
# import flowers dataset
#--------------------------------------

(train_examples, validation_examples), ds_info = tfds.load('tf_flowers',
                                                        with_info=True,
                                                        as_supervised=True,
                                                        split=['train[:70%]', 'train[70%:]']
                                                       )

num_examples = ds_info.splits['train'].num_examples
num_classes  = ds_info.features['label'].num_classes

num_train_examples      = 0
num_validation_examples = 0

for example in train_examples:
  num_train_examples += 1

for example in validation_examples:
  num_validation_examples += 1

print('Total number of classes: {}'.format(num_classes))
print('Total number of images: {}'.format(num_examples))
print('Total number of training images: {}'.format(num_train_examples))
print('Total number of validation images: {}'.format(num_validation_examples))


# once again, you can see that the images are not all the same size
# and so we'll have to resize them all
#for i, example_image in enumerate(train_examples.take(3)):
#  print("Image {} shape: {}".format(i+1, example_image[0].shape))

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches      = train_examples.shuffle(num_train_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)


#--------------------------
# check model predictions
#--------------------------

class_names = np.array(ds_info.features['label'].names)
print(class_names)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch       = flowers_sm.predict(image_batch)
predicted_batch       = tf.squeeze(predicted_batch).numpy()
predicted_ids         = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]
print(predicted_class_names)

print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  plt.title(predicted_class_names[n].title(), color=color)
  plt.axis('off')

_ = plt.suptitle("Model predictions: Flowers (MobileNet)")
plt.show()

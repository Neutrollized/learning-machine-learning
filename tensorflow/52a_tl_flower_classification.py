#!/usr/bin/env python3

#######################################################
#
# Transfer Learning (flower classifiation edition)
# - using MobileNet
# - also saving model at the end
#
#######################################################

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np

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


#----------------------------------------
# applying transfer learning
#----------------------------------------

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))

# freeze variables in feature extractor layer
feature_extractor.trainable = False


#---------------------------------------------
# attch our output/classification layer
#---------------------------------------------

model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(num_classes)
])

print(model.summary())


#------------------------
# train model
#------------------------

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)

# was initially run at 10 EPOCHS (because it was a relatively small image set)
# and saw that validation accuracy plateaued after the 4th EPOCH
# so I'm just going to set this to 6
EPOCHS = 6
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

# plot accuracy/loss graphs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#--------------------------
# check model predictions
#--------------------------

class_names = np.array(ds_info.features['label'].names)
print(class_names)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch       = model.predict(image_batch)
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


#---------------------------------------------------
# save as keras .h5 file
#---------------------------------------------------

model.save("./saved_models/flower_classification.h5")

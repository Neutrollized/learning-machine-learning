#!/usr/bin/env python3

#######################################################
#
# Cats & Dogs with (Color) CNNs
# with Image Augmentation
# and Dropout
# Using Sigmoid activation function instead of Softmax
# https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab6-Cats-v-Dogs.ipynb
#
#######################################################

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
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

# goes to ~/.keras/datasets
_URL    = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=True)

zip_dir_base = os.path.dirname(zip_dir)

base_dir       = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir      = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir      = os.path.join(train_dir, 'cats')
train_dogs_dir      = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val   = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


#----------------------
# data prep
#----------------------

BATCH_SIZE = 100
IMG_SHAPE  = 150

def plotImages(images_arr):
  fig, axes = plt.subplots(1, 5, figsize=(10,10))
  axes = axes.flatten()
  for img, ax in zip(images_arr, axes):
    ax.imshow(img)
  plt.tight_layout()
  plt.show()


'''
#--------------------------------
# image augmentation examples
#--------------------------------

# flip
image_gen        = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen   = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=train_dir,
                                                 shuffle=True,
                                                 target_size=(IMG_SHAPE,IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# rotate
image_gen        = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_data_gen   = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=train_dir,
                                                 shuffle=True,
                                                 target_size=(IMG_SHAPE,IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# zoom
image_gen        = ImageDataGenerator(rescale=1./255, zoom_range=0.5)
train_data_gen   = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=train_dir,
                                                 shuffle=True,
                                                 target_size=(IMG_SHAPE,IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
'''

# applying all sorts of augmentations 
image_gen_train = ImageDataGenerator(
  rescale=1./255,
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest'
)

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='binary')

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=validation_dir,
                                                 shuffle=False,
                                                 target_size=(IMG_SHAPE,IMG_SHAPE),
                                                 class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


#-------------------------------
# model creation
#-------------------------------

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
  tf.keras.layers.SpatialDropout2D(0.2),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.SpatialDropout2D(0.2),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.SpatialDropout2D(0.2),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.SpatialDropout2D(0.2),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())


#-------------------------
# train model
#-------------------------

EPOCHS = 20

history = model.fit(
  train_data_gen,
  steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
  epochs=EPOCHS,
  validation_data=val_data_gen,
  validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./plots/34_plot.png')
plt.show()

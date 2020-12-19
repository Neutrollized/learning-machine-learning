#!/usr/bin/env python3

from keras.datasets import mnist

from keras import models
from keras import layers

from keras.utils import to_categorical

import matplotlib.pyplot as plt


##################################################################################################

# dataset comes in the form of a set of 4 numpy arrays which we split into
# training set (images & labels)  and test set (images & labels)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# images come in as uint8, or unsigned 8-bit integers (i.e. 2^8 = 256 or [0, 255])
print(train_images.shape)
print('number of training labels:', len(train_labels))
print('training label data type:', train_labels.dtype)
print(train_labels)

# displaying the 5th digit
# https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.imshow.html
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

print(test_images.shape)
print('number of test labels:', len(test_labels))
print('test label data type:', test_labels.dtype)
print(test_labels)

# neural network architecture
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# compile
# loss function = how to measure its performance and steer itself in the right direction
# optimizer = how the network will update itself based on the data and loss function
# metrics = metrics to monitor during training & testing
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# preparing image data
# we normalize the images from int[0, 255] to float[0, 1] 
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
print('train image data type:', train_images.dtype)

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
print('test image data type:', test_images.dtype)

# preparing labels
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels)

# train ("fit" the model to its training data)
EPOCHS = 5
network.fit(train_images, train_labels, epochs=EPOCHS, batch_size=128)

# https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

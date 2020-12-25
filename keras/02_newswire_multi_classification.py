#!/usr/bin/env python3

from keras.datasets import reuters

from keras import models
from keras import layers

from keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np


##################################################################################################

# the argument num_words means you'll keep only the top 10k most frequently
# occurring words in the training data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print(train_data.shape)
print(test_data.shape)

# each review (train_data) is a list of word indices
print(train_data[0])
# and because you're restricting yourself to the top 10k words, no index will exceed 10k
print("max index:", max([max(sequence) for sequence in train_data]))

# and here's how you would decode a review back to English
word_index = reuters.get_word_index()
#print(word_index)

reverse_word_index = dict(
  [(value, key) for (key, value) in word_index.items()])
# indices are offset by 3 becuase 0, 1, 2 contain metadata
decoded_review = ' '.join(
  [reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)

# there are 46 different topics
# each label (0-45) is associated with a different topic
print(train_labels[0])


#----------------------------------
# preparing the data
#----------------------------------

# this function one-hot encodes the sequence of word indices in a review
# for example train_data[0] will have its indices one-hot encoded
# turning it into a 10k-dimensional vector of 0's and the indices of the
# words used would be 1's
def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

x_train = vectorize_sequences(train_data)
x_test  = vectorize_sequences(test_data)


def one_hot(labels, dimension=46):
  results = np.zeros((len(labels), dimension))
  for i, label in enumerate(labels):
    results[i, label] = 1.
  return results

one_hot_train_labels = one_hot(train_labels)
one_hot_test_labels  = one_hot(test_labels)

print(one_hot_train_labels[0])


#---------------------------
# building model
#---------------------------

network = models.Sequential()
network.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(46, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# shape is (8982,)
partial_x_train = x_train[:-1000]
x_val           = x_train[-1000:]
partial_y_train = one_hot_train_labels[:-1000]
y_val           = one_hot_train_labels[-1000:]


#-----------------------
# training model
#-----------------------

EPOCHS = 20
history = network.fit(partial_x_train,
                      partial_y_train,
                      epochs=EPOCHS,
                      batch_size=256,
                      validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

# bo = blue dot, b = blue line
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.clf() # clears the figure
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#------------------------------
# rerun with less epochs
#------------------------------

network.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512)
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
results = network.evaluate(x_test, one_hot_test_labels)
print(results[1])	# returns loss and metric (accuracy) values

# apply trained network to test data to see the likely hood it being each of the 46 categories
prediction = network.predict(x_test)
#print(prediction)

# because each row is a different probablility vector for each piece of input data
# here we'll print the winning probability (confidence) percentage of each row
#
# NOTE: 0 should refer to the rows and 1 should refer to the columns,
#       so why is axis=1 here to get the max value of each row?
#       think of it as .sum(axis=0) which sums along the rows (producing column totals)
#       (it's fucked up, I know...)
print(np.amax(prediction, axis=1))

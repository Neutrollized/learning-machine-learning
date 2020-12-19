#!/usr/bin/env python3

from keras.datasets import imdb

from keras import models
from keras import layers

from keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np


##################################################################################################

# the argument num_words means you'll keep only the top 10k most frequently
# occurring words in the training data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data.shape)

# each review (train_data) is a list of word indices
print(train_data[0])
# and because you're restricting yourself to the top 10k words, no index will exceed 10k
print("max index:", max([max(sequence) for sequence in train_data]))

# and here's how you would decode a review back to English
word_index = imdb.get_word_index()
#print(word_index)

reverse_word_index = dict(
  [(value, key) for (key, value) in word_index.items()])
# indices are offset by 3 becuase 0, 1, 2 contain metadata
decoded_review = ' '.join(
  [reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)

# each label is a 0 negative review, 1 is a positive
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

print(x_train[0])

y_train = np.asarray(train_labels).astype('float32')
y_test  = np.asarray(test_labels).astype('float32')


#---------------------------
# building model
#---------------------------

network = models.Sequential()
network.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

# shape is (25000,)
# we're going to train on the first 15k (i.e [:-10000] = beginning up until the last 10k)
# the last 10k will be used for validation (i.e. [-10000:] = starting from 10k from the end up until the end
partial_x_train = x_train[:-10000]
x_val           = x_train[-10000:]
partial_y_train = y_train[:-10000]
y_val           = y_train[-10000:]

'''
# another way to do the same thing but perhaps in an easier to read format is..
# take your first 10k as the validation, and the left over for training
# it doesn't matter if you make the head or tail of your dataset as the validation set
# what matters is how you split them (commonly 70:30 training:validation split)
x_val           = x_train[:10000]
partial_x_train = x_train[10000:]
y_val           = y_train[:10000]
partial_y_train = y_train[10000:]
'''


#-----------------------
# training model
#-----------------------

EPOCHS = 20
history = network.fit(partial_x_train,
                      partial_y_train,
                      epochs=EPOCHS,
                      batch_size=512,
                      validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

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

network.fit(x_train, y_train, epochs=4, batch_size=512)
results = network.evaluate(x_test, y_test)
print(results[1])

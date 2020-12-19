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


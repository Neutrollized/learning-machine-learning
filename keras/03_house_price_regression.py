#!/usr/bin/env python3

from keras.datasets import boston_housing

from keras import models
from keras import layers

from keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np


##################################################################################################

# the argument num_words means you'll keep only the top 10k most frequently
# occurring words in the training data
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

print(train_data[0])
print(train_labels[0])  # price of house in 1000's (this is in mid-1970s)


#----------------------------
# preparing the data
#----------------------------

mean = train_data.mean(axis=0)
print("mean:", mean)
train_data -= mean
std = train_data.std(axis=0)
print("std deviation:", std)
train_data /= std
test_data -= mean
test_data /= std
# data should be normalized now
print("normalized training data:", train_data)
print("normalized test data:", test_data)


#------------------------
# building model
#------------------------

def build_model():
  model = models.Sequential()
  model.add(layers.Dense(64, activation='relu',
                         input_shape=(train_data.shape[1],)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1))
  model.compile(optimizer='rmsprop',
                loss='mse',
                metrics=['mae'])
  return model

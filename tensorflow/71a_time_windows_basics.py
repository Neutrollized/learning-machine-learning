#!/usr/bin/env python3

##################################################
#
# Time Series Forecasting (time windows)
#
##################################################

import tensorflow as tf
from tensorflow import keras

import numpy as np


#-----------------------------------------
# building it up incrementally...
#-----------------------------------------

# create a dataset of 0-9
# window them in size 5, shift 1
# there are some short windows as you run out of range
print("\nwindowed and shifted data:")
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1)
for window_dataset in dataset:
  for val in window_dataset:
    print(val.numpy(), end=" ")
  print()

# like the previous except short windows are dropped
print("\nno short windows:")
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
for window_dataset in dataset:
  for val in window_dataset:
    print(val.numpy(), end=" ")
  print()

# ML models will want the data represented as tensors (vector)  and not datasets
# hence using flat_map function to convert each windowed dataset into a tensor
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#flat_map
print("\nrepresented as tensors:")
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
  print(window.numpy())

# need input features and target to be used by ML models
# using map function to map the window tensor in to input and target tensors
# where the target is just the a tensor of size 1 (last value)
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map
print("\nsplit into input and target tensors:")
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x, y in dataset:
  print(x.numpy(), y.numpy())

# IID = identical and independely distributed
# shuffle the training batch so that they contains fairly independent windows
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
print("\nshuffle (since IID):")
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
for x, y in dataset:
  print(x.numpy(), y.numpy())

# in a ML model you, you provide batches
# typically batch size of 32
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch
print("\nprovide batches of instances:")
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())


#-----------------------------------
# putting it all together...
#-----------------------------------

# from_tensor_slices methods creates dataset
# where each value corresponds to 1 time step from the time series
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
def window_dataset(series, window_size, batch_size=32):
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size + 1,
                 shift=1,
                 drop_remainder=True
                )
  ds = ds.flat_map(lambda w: w.batch(window_size + 1))
  ds = ds.map(lamba w: (w[:-1], w[-1]))
  ds = ds.shuffle(len(series))
  return ds.batch(batch_size).prefetch(1)

#!/usr/bin/env python3

##################################################
#
# Time Series Forecasting (time windows)
#
##################################################

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np


#--------------------
# setup
#--------------------
def plot_series(time, series, format="-", start=0, end=None, label=None):
  """
  time(x) = 1d array; values are the time steps (i.e. 0, 1, 2, ...)
  series(y) = 1d array; the time series(y) values at the x (we will use functions to generate this)
  """
  plt.plot(time[start:end], series[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("Value")
  if label:
    plt.legend(fontsize=14)
  plt.grid(True)

def trend(time, slope=0):
  return slope * time

def seasonal_pattern(season_time):
  """Just an artibrary pattern"""
  return np.where(season_time < 0.4,
                  np.cos(season_time * 2 * np.pi),
                  1 / np.exp(3 * season_time)
                 )

def seasonality(time, period, amplitude=1, phase=0):
  """Repeats the same pattern at each period"""
  season_time = ((time + phase) % period) /period
  return amplitude * seasonal_pattern(season_time)

def white_noise(time, noise_level=1, seed=None):
  rnd = np.random.RandomState(seed)
  return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1)

slope     = 0.05
baseline  = 10
amplitude = 40
series    = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

noise_level = 5
noise       = white_noise(time, noise_level, seed=42)

series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


#-----------------------------------------
# forecasting with machine learning
#-----------------------------------------

def window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1,
                           shift=1,
                           drop_remainder=True
                          )
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.shuffle(shuffle_buffer)
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

split_time = 1000
time_train = time[:split_time]
x_train    = series[:split_time]
time_valid = time[split_time:]
x_valid    = series[split_time:]


#-----------------------
# linear model
#-----------------------

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set = window_dataset(x_train, window_size)
valid_set = window_dataset(x_valid, window_size)

# we're forecasting a single value, we just need a single unit
model = keras.models.Sequential([
  keras.layers.Dense(1, input_shape=[window_size])
])

# SGD = stochastic gradient descent
# https://keras.io/api/optimizers/sgd/
optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)

# Huber is quadratic for small errors (i.e. MSE), linear for large ones (i.e. MAE)
# it's a good function to use when you want to optimize MAE
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
early_stopping = keras.callbacks.EarlyStopping(patience=10)

# we can set the epochs to be high as early stopping callback will likely
# have stopped it long before that
EPOCHS = 500
model.fit(train_set,
          epochs=EPOCHS,
          validation_data=valid_set,
          callbacks=[early_stopping]
         )



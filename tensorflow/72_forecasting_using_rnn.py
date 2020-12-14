#!/usr/bin/env python3

###################################################################
#
# Time Series Forecasting using Recurrent Neural Networks (RNN)
#
###################################################################

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


split_time = 1000
time_train = time[:split_time]
x_train    = series[:split_time]
time_valid = time[split_time:]
x_valid    = series[split_time:]


#------------------------------------
# stateful RNN forecasting
#------------------------------------

def sequential_window_dataset(series, window_size):
  series = tf.expand_dims(series, axis=-1)
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size + 1,
                 shift=window_size,
                 drop_remainder=True
                )
  ds = ds.flat_map(lambda window: window.batch(window_size + 1))
  ds = ds.map(lambda window: (window[:-1], window[1:]))
  return ds.batch(1).prefetch(1)


for x_batch, y_batch in sequential_window_dataset(tf.range(10), 3):
  print(x_batch.numpy(), y_batch.numpy())


class ResetStatesCallback(keras.callbacks.Callback):
  def on_epoch_begin(self, epoch, logs):
    self.model.reset_states()


#-------------------------------------------------
# statefun RNN (sequence-to-vector example)
#-------------------------------------------------

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set   = sequential_window_dataset(x_train, window_size)

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
# return_sequences defaults to False and will return only the final output vector (sequence-to-vector)
# but what we want is for the first RNN layer to feed into the second so it's not just the final output
#
# the lambda at the end scales up the output by a factor of 200 to help training
# because the default activation function is "hypberbolic tangent (tanh)", the return value is between -1 and 1
model = keras.models.Sequential([
  keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
  keras.layers.SimpleRNN(100, return_sequences=True),
  keras.layers.SimpleRNN(100),
  keras.layers.Dense(1),
  keras.layers.Lambda(lambda x: x * 200.0)
])

reset_states = ResetStatesCallback()

optimizer = keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

model.fit(train_set, epochs=150, callbacks=[reset_states])

s2v_forecast = model.predict(series[np.newaxis, :, np.newaxis])
print(s2v_forecast)


#-------------------------------------------------
# stateful RNN - finding a good learning rate
#-------------------------------------------------

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set   = sequential_window_dataset(x_train, window_size)

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
# input shape is 3d: batch size, time step, dimensionality of input sequence
# number of time steps here is defined as 'None', meaning it will handle sequences of any length
# this is a univariate time series and hance dimensionality of input sequence is '1'
#
# return_sequences defaults to False and will return only the final output vector (sequence-to-vector)
# but what we want is for the first RNN layer to feed into the second so it's not just the final output
# we do the same thing for the second RNN layer so that it feeds into the Dense layer
# and each returns a value, giving us a sequence as output (sequence-to-sequence)
model = keras.models.Sequential([
  keras.layers.SimpleRNN(100, return_sequences=True, stateful=True,
                         batch_input_shape=[1, None, 1]),
  keras.layers.SimpleRNN(100, return_sequences=True, stateful=True),
  keras.layers.Dense(1),
  keras.layers.Lambda(lambda x: x * 200.0)
])

lr_schedule = keras.callbacks.LearningRateScheduler(
  lambda epoch: 1e-8 * 10**(epoch / 30)
)

reset_states = ResetStatesCallback()

optimizer = keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

history = model.fit(train_set, epochs=200, callbacks=[lr_schedule, reset_states])

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])
plt.show()


#----------------------------------------
# stateful RNN - final
#----------------------------------------

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set   = sequential_window_dataset(x_train, window_size)
valid_set   = sequential_window_dataset(x_valid, window_size)

model = keras.models.Sequential([
  keras.layers.SimpleRNN(100, return_sequences=True, stateful=True,
                         batch_input_shape=[1, None, 1]),
  keras.layers.SimpleRNN(100, return_sequences=True, stateful=True),
  keras.layers.Dense(1),
  keras.layers.Lambda(lambda x: x * 200.0)
])

reset_states = ResetStatesCallback()

# setting a good learning rate is very important
# too high and you get instabilities and you don't really learn
# too low and it takes a really long time to learn
#
# loss can also be a bit unpredictable so you don't want early stopping to stop your
# training too early, and hence we set the patience to be something higher (60)
optimizer = keras.optimizers.SGD(lr=1e-7, momentum=0.9)

model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

early_stopping = keras.callbacks.EarlyStopping(patience=60)

# save the best model at each EPOCH (if it's the best one)
model_checkpoint = keras.callbacks.ModelCheckpoint(
  "./saved_models/my_72_checkpoint.h5", save_best_only=True
)

EPOCHS = 500
model.fit(train_set,
          epochs=EPOCHS,
          validation_data=valid_set,
          callbacks=[early_stopping, model_checkpoint, reset_states]
         )

# load the best model that was saved
model = keras.models.load_model("./saved_models/my_72_checkpoint.h5")

model.reset_states()
rnn_forecast = model.predict(series[np.newaxis, :, np.newaxis])
rnn_forecast = rnn_forecast[0, split_time -1:-1, 0]

print(rnn_forecast.shape)

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
plt.show()

print(keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())

#!/usr/bin/env python3

##################################################
#
# Time Series Forecasting
# - just showing some common patterns
# - metrics options
#
# TODO: need to add comments and references 
#       for better understanding
#
##################################################

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np


#-------------
# setup
#-------------

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


#--------------------------------
# trend & seasonality
#--------------------------------

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


#---------------------------------
# naive forcasting
#---------------------------------

split_time = 1000
time_train = time[:split_time]
x_train    = series[:split_time]
time_valid = time[split_time:]
x_valid    = series[split_time:]

naive_forecast = series[split_time - 1:-1]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, naive_forecast, label="Forecast")
plt.show()

print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())


#--------------------------
# moving average
#--------------------------

'''
def moving_average_forecast(series, window_size):
  """
  Forecasts the mean of the last few values.
  If window_size=1, then this is equivalent to naive forecast.
  """
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast)
'''

def moving_average_forecast(series, window_size):
  """
  Forecasts the mean of the last few values.
  If window_size=1, then this is equivalent to naive forecast.
  This implementation is muich, much faster than the previous one,
  but not as straigh-forward to understand how it works.
  """
  # https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
  mov = np.cumsum(series)
  mov[window_size:] = mov[window_size:] - mov[:-window_size]
  return mov[window_size - 1:-1] / window_size

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, moving_avg, label="Moving Average (30 days)")
plt.show()

# the MAE here is higher compared to Naive Forecasting
# when the changes are small, the Moving Average is pretty good,
# but it's spikes/large changes (i.e. due to seasonality) that the 
# Moving Average needs a bit more time to pick up on
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())


#----------------------
# differencing
#----------------------

diff_series = (series[365:] - series[:-365])
diff_time   = time[365:]

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series, label="Series(t) - Series(t-365)")
plt.show()

# focusing on the validation perid
plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:], label="Series(t) - Series(t-365)")
plt.show()


# moving avearge of differenced series
# the window used here is 50, but that's a param that can be tuned
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:], label="Series(t) - Series(t-365)")
plot_series(time_valid, diff_moving_avg, label="Moving Average of Diff")
plt.show()


# moving average of differenced series + past values
diff_moving_avg_plus_past = diff_moving_avg + series[split_time - 365:-365]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, diff_moving_avg_plus_past, label="Forecasts (noisy)")
plt.show()

print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())


# moving average on past values
diff_moving_avg_plus_smooth_past = diff_moving_avg + moving_average_forecast(series[split_time - 370:-359], 11)

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, diff_moving_avg_plus_smooth_past, label="Forecasts (smooth)")
plt.show()

print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())

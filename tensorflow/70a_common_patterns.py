#!/usr/bin/env python3

##################################################
#
# Time Series Forecasting
# - no tf stuff in this one
# - just showing some common patterns
#
##################################################

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


#-----------------------------------------
# generate an upward trending pattern 
#-----------------------------------------

def trend(time, slope=0):
  return slope * time


time = np.arange(4 * 365 + 1)
baseline = 10
series = baseline + trend(time, 0.1)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

print(time)
print(series)


#------------------------------------------------
# generate a time series w/seasonal pattern
#------------------------------------------------

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


amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


#--------------------------------
# both trend & seasonality
#--------------------------------

slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

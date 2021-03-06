#!/usr/bin/env python3

####################################
#
# Celcius to Fahrenheit
# https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb
#
####################################

import tensorflow as tf
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))


# same as 
#model = tf.keras.Sequential([
#  tf.keras.layers.Dense(units=1, input_shape=[1])
#])
# same as below:
layer0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer0])

# list the weights and bias for each input column (weight)
# here, it's just 1 w and 1 b (always just 1 bias for each dense layer)
w, b = model.weights
print("Weights\n{}\nBias\n{}".format(w, b))

# Adam stands for "ADAptive with Momentum"
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")


#--------------------------------

import matplotlib.pyplot as plt

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])

# display the plot (you don't need this if it was in a notebook)
plt.show()

print(model.predict([100.0]))

print("These are the layer variables: {}".format(layer0.get_weights()))

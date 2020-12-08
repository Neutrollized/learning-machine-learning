#!/usr/bin/env python3

####################################
#
# Celcius to Fahrenheit
# https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb
#
###################################

import tensorflow as tf
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

layer0 = tf.keras.layers.Dense(units=4, input_shape=[1])
layer1 = tf.keras.layers.Dense(units=4)
layer2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([layer0, layer1, layer2])

#---------------------------------------------------------------------------------
# NOTE: as you add more layers, the way each layer gets passed to the next
# is very much like multiplying matrices 
# 1 x 4 
w0, b0 = layer0.weights
print("Weights\n{}\nBias\n{}".format(w0, b0))

# layer0 was 4 units, which gets passed to layer1 at 4 units EACH
# hence this layer will show as a 4 x 4
w1, b1 = layer1.weights
print("Weights\n{}\nBias\n{}".format(w1, b1))

# layer1 was a 4 x 4, which gets fed into layer2 which is a 4 x 1
# and results in a 4 x 1
w2, b2 = layer2.weights
print("Weights\n{}\nBias\n{}".format(w2, b2))
#---------------------------------------------------------------------------------

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
print("These are the layer variables: {}".format(layer1.get_weights()))
print("These are the layer variables: {}".format(layer2.get_weights()))

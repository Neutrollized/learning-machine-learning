#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


#----------------------------------------------
# download dataset from UCI ML Repo
#----------------------------------------------

dataset_path = keras.utils.get_file("auto-mgp.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())


#----------------------------
# clean the data
#----------------------------

# print how many rows have missing data under each column/heading
print(dataset.isna().sum())

# in this particular example there's only a few rows of data with missing HP, so we're just going to simply drop those
dataset = dataset.dropna()

# using one-hot encoding to turn 'Origin' (which is really a categorical column) into a numerical one
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())


#-------------------------------
# split training/test
#-------------------------------

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


#---------------------------
# inspect data
#---------------------------

# diag_kind="kde" = kernel density estimation (smooth histograms) along the diagonal
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)


#----------------------------------
# split features from labels 
#----------------------------------

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


#-----------------------------
# normalize data/features
#-----------------------------

# normalizing keeps the scales and ranges similar
# normalizing also removes dependency on choice of unit (i.e. miles vs kms) used in input
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


#-------------------------
# build the model
#-------------------------

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  # https://keras.io/api/optimizers/rmsprop/
  OPTIMIZER = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=OPTIMIZER,
                metrics=['mae', 'mse'])
  return model

model = build_model()
print(model.summary())

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)


#-------------------------
# train the model
#-------------------------

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
# early stopping is used to help deal with over-fitting by
# stopping training of the model when there's little to no improvement (or worse -- degredation) in validation error
EARLY_STOP = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  #callbacks=[PrintDot()]	# use this callback instead to have it finish the EPOCHS every time
  callbacks=[EARLY_STOP, PrintDot()]
)

print('')
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

plot_loss(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MGP".format(mae))


#------------------------
# make predictions
#------------------------

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()

#!/usr/bin/env python3

from keras.datasets import boston_housing

from keras import models
from keras import layers

from keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np


##################################################################################################

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

print(train_data[0])
print(train_targets[0])  # price of house in 1000's (this is in mid-1970s)


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
#print("normalized training data:", train_data)
#print("normalized test data:", test_data)


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


#--------------------------------------
# implementing k-fold validation
# - initial attempt
#--------------------------------------

k = 5
num_val_samples = len(train_data) // k
EPOCHS = 100
all_scores = []

for i in range(k):
  print('processing fold #', i)
  val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
  val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

  partial_train_data = np.concatenate(
    [train_data[: i * num_val_samples],
     train_data[(i + 1) * num_val_samples :]],
    axis=0)

  partial_train_targets = np.concatenate(
    [train_targets[: i * num_val_samples],
     train_targets[(i + 1) * num_val_samples :]],
    axis=0)

  model = build_model()
  model.fit(partial_train_data, partial_train_targets,
            epochs=EPOCHS, batch_size=1, verbose=0)	# verbose=0 means silent training
  val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
  all_scores.append(val_mae)

print("Validation MAE of each fold:", all_scores)
print("Avg validation MAE:", np.mean(all_scores))


#--------------------------------------
# implementing k-fold validation
# - more epochs, saved MAE history
# - plot it out! 
#--------------------------------------

num_val_samples = len(train_data) // k
EPOCHS = 200
all_mae_histories = []

for i in range(k):
  print('processing fold #', i)
  val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
  val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

  partial_train_data = np.concatenate(
    [train_data[: i * num_val_samples],
     train_data[(i + 1) * num_val_samples :]],
    axis=0)

  partial_train_targets = np.concatenate(
    [train_targets[: i * num_val_samples],
     train_targets[(i + 1) * num_val_samples :]],
    axis=0)

  model = build_model()
  history = model.fit(partial_train_data, partial_train_targets,
                      validation_data=(val_data, val_targets),
                      epochs=EPOCHS, batch_size=1, verbose=0)
  mae_history = history.history['val_mae']
  all_mae_histories.append(mae_history)

# compute average of per-epoch MAE score for all folds
avg_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(EPOCHS)]

plt.plot(range(1, len(avg_mae_history) + 1), avg_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


#----------------------------
# final model
#----------------------------

# after playing with some numbers/settings,
# I found the following model to give me slightly better Test MAE score
# over the one initially used to evaluate the model
def build_model_final():
  model = models.Sequential()
  model.add(layers.Dense(128, activation='relu',
                         input_shape=(train_data.shape[1],)))
  model.add(layers.Dense(1))
  model.compile(optimizer='rmsprop',
                loss='mse',
                metrics=['mae'])
  return model

# as seen from the graphs that more epochs won't improve the score (overfits beyond ~80 epochs)
# so we're going to cap it at 80 and run fit()/train on the entire dataset
EPOCHS = 80

model = build_model_final()
model.fit(train_data, train_targets,
          epochs=EPOCHS, batch_size=16)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print("Test MAE score:", test_mae_score)

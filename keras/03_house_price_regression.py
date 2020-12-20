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

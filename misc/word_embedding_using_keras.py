import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.datasets import reuters
from keras.preprocessing import sequence

num_words = 1000


(reuters_train_x, reuters_train_y), (reuters_test_x, reuters_test_y) = tf.keras.datasets.reuters.load_data(num_words=num_words)
n_labels = np.unique(reuters_train_y).shape[0]

#print(n_labels)

#-----------------------------------------

#from keras.utils import np_utils
from keras.utils import to_categorical

#reuters_train_y = np_utils.to_categorical(reuters_train_y, 46)
#reuters_test_y = np_utils.to_categorical(reuters_test_y, 46)
reuters_train_y = to_categorical(reuters_train_y, 46)
reuters_test_y = to_categorical(reuters_test_y, 46)

reuters_train_x = tf.keras.preprocessing.sequence.pad_sequences(reuters_train_x, maxlen=20)
reuters_test_x = tf.keras.preprocessing.sequence.pad_sequences(reuters_test_x, maxlen=20)

#-----------------------------------------

from tensorflow.keras import layers

model = tf.keras.Sequential(
  [
    layers.Embedding(num_words, 1000, input_length=20),
    layers.Flatten(),
    layers.Dense(256),
    layers.Dropout(0.25),
    layers.Activation('relu'),
    layers.Dense(46),
    layers.Activation('softmax')
  ])


model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])

model_1 = model.fit(reuters_train_x, reuters_train_y,
                    validation_data=(reuters_test_x, reuters_test_y),
                    batch_size=128, epochs=20, verbose=0)


import matplotlib.pyplot as plt

acc = model_1.history['accuracy']
val_acc = model_1.history['val_accuracy']
loss = model_1.history['loss']
val_loss = model_1.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


plt.show()
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower left')
plt.show()

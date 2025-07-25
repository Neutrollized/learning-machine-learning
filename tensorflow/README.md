# README
The material in this folder is largely based on Udacity's [Intro to TensorFlow for Deep Learning](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187) free course + my own notes/understanding.

If you're running these from a MacOS (as I am) or Linux, you will need to compile your own TensorFlow as the default package you install via pip currently won't have a lot of CPU extensions that will speed up your CPU-only training.  I have outlined some of the steps I took to compiling my own [**here**](https://gist.github.com/Neutrollized/6a409146fcf02438852c27633a031d0f)

I also recommend using [`pyenv`](https://realpython.com/intro-to-pyenv/) to setup your, uh, Python environment so you don't have to reinstall all your packages after an upgrade or whatnot.

## Dependencies
- **tensorflow** (it will install other package requirements, including **numpy**)
- **matplotlib**
- **scikit-learn**
- [**tensorflow-datasets**](https://blog.tensorflow.org/2019/02/introducing-tensorflow-datasets.html)
- [**seaborn**](https://seaborn.pydata.org/) (Auto MPG example)
- [**tensorflow_hub**](https://tfhub.dev/)
- **opencv-python** (`import cv2`)

`pip3 install tensorflow matplotlib scikit-learn tensorflow-datasets seaborn`

## NOTES
In some of the examples, you might see the following:
```
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
```
which is more for Jupyter notebooks as chances are you'll be running the cells multiple times.  I don't use notebooks right now as I prefer `vim` and terminal.

## 0x - Celcius to Fahrenheit (Regression)
- output: Number
- loss: Mean squared error (MSE)
- last layer activation fucntion: None


## 1x - Fashion MNIST (Classification)
Uses greyscale images of the same size
- output: Probability distribution of its confidence of a certain class
- loss: Sparse categorical crossentropy
- last layer activation fucntion: Softmax

Here's a good [CNN primer](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) used in one of the variations of this classification example

#### Exercises
Try the following settings to see the impact of it on the results/accuracy:
- set epochs = 1
- set number of neurons in Dense layer to range of something low (e.g. 10) to something high (e.g. 512)
- add additional dense layers
- don't normalize px values 


## 2x - Auto MPG
[YouTube link](https://www.youtube.com/watch?v=-vHQub0NXI4&ab_channel=TensorFlow)

## 3x - Cats & Dogs (Classification...in color!)
This is where CNNs shine because it can split/isolate the image by color (RGB).  Here we use color images of various sizes, so here we will resize all the images and each.  

For greyscale images (e.g. of size 20px x 20px), it is represented by an array of (20, 20, 1) for (height, width, color channels), where as for a color image of the same size, it would be represented by an array of (20, 20, 3) for  because the number color channels is now 3 (R/G/B) instead of 1 (greyscale)

Uses [Keras Preprocessing ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) to handle the following data prep actions:
- read images from disk
- decode image content & convert into proper grid format as per RGB content
- convert to floating point tensors
- rescale tensors from (0, 255) to (0, 1) as neural networks prefer to deal with smaller input values (i.e. normalizing)

### Reducing Over-fitting in Image Classification
A sign of over-fitting can be see when training accuracy continues to improve while validation accuracy tapers/levels of after some number of epochs (you can see this in the various *3x_plot.png*s where the training and validation accuracy diverges).  This means that the neural net is memorizing the images and not generalizing enough.  More techniques doesn't necessarily mean better as you can see from the plots that CNNs with Image Augmentation provided the best model.

#### Image Augmentation
One way you can improve on this is to have various types of images of your subject (i.e. dog at different angles/zoom levels).  If you *don't* have the luxury of a large dataset with a wealth of types of images, then you can consider [**image augmentation**](https://www.tensorflow.org/tutorials/images/data_augmentation) to apply random transformations to your images such as:
- rotate
- flip
- zoom
- ...

This will give your CNN a better chance a learning to generalize better if it has see more examples (even if some of these examples are just augmentations of ones it has already seen)

#### Dropout
With each pass through the epoch, the weights and biases in each neuron is adjusted, but some might be used more and adjusted more while others are used less, and over time, the neurons with the heavier weights and biases influence the overall outcome of the training more than others.

The way dropout solves this is by randomly turning off neurons with each pass and thereby giving the other neurons a chance to "pick up the slack".


## 4x - Flower Classification
[Exercise](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c03_exercise_flowers_with_data_augmentation.ipynb)

[Solution](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c04_exercise_flowers_with_data_augmentation_solution.ipynb)


## 5x - Transfer Learning
Instead of training your own models with large datasets (especially images), you can use a model that some 3rd party large neural networks have trained and thus transfering what they learned into your own work and improving accuracy.

You will have to change the output layer of the pre-trained model so that it matches the output you need.  You will also make sure that you don't change the pre-trained part of the pre-trained model (called "freezing") or else the features that it learned will change.

[TensorFlow Hub](https://tfhub.dev/) has a lot of pre-trained models that already has the output/classification layer stripped.  Using it is easy; here's an example:
```
CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
IMAGE_RES = 224

model = tf.keras.Sequential([
  hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3)),
  layers.Dense(2, activation='softmax')
])
```
where the trailing number (`4` in the example), is the version number.

Transfer Learning is a very important/useful concept in ML as it leverages models that were trained on very large neural nets and data sets -- something that would be both costly and time consuming if you were to do it on your own (just imagine trying to aquire a gazillion cat and dog pictures to improve your model's accuracy) 

[Transfer Learning with Flowers Classification - Exercise](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c02_exercise_flowers_with_transfer_learning.ipynb)

[Transfer Learning with Flowers Classification - Solution](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c03_exercise_flowers_with_transfer_learning_solution.ipynb)

### MobileNet vs Inception
We use MobileNet as it is very lightweight/ideal for devices with limited memory & resources (i.e. laptop), but I also have another flower classification example, **52b_tl_flower_classification.py**, that uses Inception instead.  If you compare Inception's classification model, you will see that the size of the model is larger (i.e. more layers?) but also the image resolution size is a bit larger...although, this should also give us slightly better accuracy at the cost of speed.  

For me, training on my Intel MBPro at 6 EPOCHS (to match what's run on the MobileNet version) took ~5x longer and accuracy was the same.  Even at 10 EPOCHS, the validation accuracy seems to have tapered off at ~90% on this model/dataset, which is what we got from MobileNet.


## 6x - Saving & Loading Models
For small/quick models, you can afford to train anew every time, but for large models that may take days or even weeks to train, it's very impractical to have to do it all from scratch everytime.  By saving a model, it allows you to reuse that training again later on (which is essentially Transfer Learning).  You can also load the model and continue training with additional data.


## 7x - Time Series Forecasting
A major use for ML/deep learning is time series forcasting (stock market, weather, etc.).  Honestly, Time Series Forecasting is a whole course on its own, because while it uses a lot of the ML modeling concepts previously discussed, it has its own unique challenges and things to consider as it deals with time which is constantly change (as opposed to training on a tonne of images of dogs and dog breeds).

### Measuring Performance
#### Fixed Partitioning
Split model into:
- training period
- validation period
- test period
For time series with some seasonality, you generally want to ensure that each period contains a whole number of seasons -- what that means is you want 1 year, 2 years, 3 years, etc. and not 1.5 years because then some seasons will get a higher representation (i.e. 2 summers, but only 1 winter).  

As for how the periods are used, you would generally train on the training period and validate with the validation period.  After you've obtained what you believe to be your best model, you would retrain on training + validation and validate with the test period to get an idea of how well your model might perform in production.  Finally, you will want to do one last training on training + validation + test before actually deploying it in prod.

Why do we train on the test set?  Because it's a time series, having the most recent data is relevant (vs a flower classification model which time is not really a factor).  As such, quite often you'll see models with just a training and validation period since the test period is in the future.

#### Roll-Forward Partitioning
Start with a short training period and increase it incrementally (+1 day or +1 week at a time).  Drawback is it requires more training time, but the benefit is that it more closely mimics real life as you get (daily) updated data.  IRL, you wouldn't want to create a time series forecasting model that was built on last year's data and still use it 2 years down the road -- you'd want to update your model regularly.

### Naive Forecasting
Take the last value and assume the next value will be the same (think of it like a shift to the right on the x-axis).  Useful to measure the performance of naive forecasting for establishing a baseline.

*"Tomorrow is going to be the same as today"*

### Moving Average
Taking the mean value of over a shorter period of time (i.e. last 30 days).  The pro here is that it's removes a lot of the noise as it focuses on a smaller window and you get a smoother forecast...which leads to the con of it not being able to anticipate trend or seasonality and so it quite often performs a little worse than Naive Forecasting.

*"Tomorrow is going to be closer to the average of the last 30 days"*

#### Differencing
One way to overcome the weakness of the Moving Average is by removing the trend and seaonality from the time series altogether by using [Differencing](https://towardsdatascience.com/an-intuitive-guide-to-differencing-time-series-in-python-1d6c7a2c067a) whereby instead of analyzing the time series itself, you analyze the difference between the time series (t) with itself a year (i.e. t-365) which will preduce a time series without the trend and seasonality.  This, of course only gets a forecast of the differenced series and not the original series, so you restore back the series from a year ago that you subtracted (and thus restoring the trend and seasonality as well).

i.e. **Forecasts = Moving Average of Differenced series + series(t-365)**

However, this forecast still has a lot of noise, which you can remove by applying the Moving Average on the past series as well.

i.e. **Forecasts = trailing Moving Average of Differenced series + centered Moving Average of past series(t-365)**

#### Trailing vs Centered Windows
Let's say we're gonna use a a 30 day window as an example.  A trailing is as you would expect -- the last 30 days.  A centered window will take into account the few days prior and after the window.  A centered window is more accurate than a trailing window, which is why we use it when calculating the Moving Average in the past series.  However, when calculating the the Moving Average of the present, we can only use a trailing window because we're not able to predict the future. 

### Recurrent Neural Networks (RNN)
An RNN is a neural network that contains a recurrent layer.  A recurrent layer is one that sequentially process a sequence of inputs.  Here's an example of what that means...

Consider the following sentences:

**"Queen B is my favorite singer."**

**"The bees died to protect their queen."**

**"Queen takes pawn."**

**"I got knighted by the queen."**

As you read word by word in the sentences, you're updating the what "queen" refers to based on the context of the sentence.  Your brain is an RNN in this case sequentially processing each input (word).  That's how you know they're all referring to different queens (Beyonce, bees, chess, monarchy).  As you can probobly guess by now that RNNs are used most often in natural language processing (NLP), but you can also use it in time series forecasting, which will be the case here.

#### How it works
So normally, you take an input time step (x0), and produce and output value for that time step (y0).  In addition to that, it also produces a state/context vector (h0), which gets fed as an additonal input for the next time step (x1).

i.e.
- x0 --> y0 + h0
- h0 + x1 --> y1 + h1
- h1 + x2 --> y2 + h2
- ...
- h29 + x30 --> y30 + h30

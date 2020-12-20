# README
The majority of the content in this folder are examples taken from [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) as [recommended reading](https://www.tensorflow.org/resources/learn-ml) from TensorFlow's site.  This book focuses on Deep Learning and has a lot of [Keras](https://keras.io/) examples; you may see a lot of overlap with content in the other folders as Keras is a popular library used in ML.

## Dependencies
- **keras**

## Data Representation in Neural Networks
Data is stored in multi-dimensional Numpy arrays called *tensors*.  It almost always contains numerical data. 

Tensors have 3 key attributes:
- **number of axes (rank)** - you can call the `ndim` function to find the rank
- **shape** - a tuple of integers that describe how many dimensions the tensor has at each axis
- **data type (dtype)**

Below are some common names and examples:

### Scalars (0D tensors)
```
import numpy as np
x = np.array(12)
print(x)
print(x.shape)
print(x.ndim)
```
Outputs:
```
12
()
0
```

### Vectors (1D tensors)
Vectors only have 1 axis, hence the shape of the example below is `(4,)`.
```
import numpy as np
x = np.array([12, 3, 6, 14])
print(x)
print(x.shape)
print(x.ndim)
```
Outputs:
```
[12  3  6 14]
(4,)
1
```

### Matrices (2D tensors)
You tend to see a lot of 2D tensors, i.e. `(samples, features)`.  An example would be a dataset of 10000 people characterized as a vector of 3 values: age, postal code, income -- and so the entire dataset can be stored in a 2D tensor of shape (10000, 3).
```
import numpy as np
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x)
print(x.shape)
print(x.ndim)
```
Outputs:
```
[[5 78  2 34  0]
 [6 79  3 35  1]
 [7 80  4 36  2]]
(3, 5)
2
```

### 3D (and higher) tensors
Time series or sequence data tend to be 3D, i.e. `(samples, timesteps, features)`.  The time axis is always the second axis (axis index=1), by convention.  For example, if you're tracking stock prices every minute: current, high, low -- for 8 hours a day or 480 minutes, and 250 days a year, you would get a time series of (250, 480, 3), where each sample would be one day's worth of data. 

Images are generally 4D, i.e. `(samples, height, width, channels)`.  Greyscale images have just 1 color channel, where as colored images would have 3 (R/G/B).  So 100 greyscale images that are 128 x 128 px would be (100, 128, 128. 1), where as 200 color images that are 256 x 256 px in size would be represented with with (200, 256, 256, 3).

Video are 5D, i.e. `(samples, frames, height, width, channels)`.  To understand video data, you have to think about it as a series of frames where eeach frame is a 3D tensor.  So a a sequence of frames (i.e. 1 video) would be 4D.  So a bunch of videos would be 5D.  A 60 second, 144 x 256 YT video sampled at 4 frames per seconds would have 240 frames, and if you were to have a batch of 10 such videos, then the 5D tensor would be (10, 240, 144, 256, 3).
```
import numpy as np
x = np.array([[[1, 78, 5, 34, 0],
               [2, 79, 6, 35, 1]],
              [[4, 81, 8, 34, 0],
               [5, 82, 9, 35, 1]],
              [[7, 84, 1, 34, 0],
               [8, 85, 2, 35, 1]]])
print(x)
print(x.shape)
print(x.ndim)
```
Outputs:
```
[[[ 1 78  5 34  0]
  [ 2 79  6 35  1]]

 [[ 4 81  8 34  0]
  [ 5 82  9 35  1]]

 [[ 7 84  1 34  0]
  [ 8 85  2 35  1]]]
(3, 2, 5)
3
```


## 00 - Recognizing Handwritten Digits
The "Hello world" of deep learning with Keras.  Taken from the MNIST dataset that comes with Keras
- classify greyscale images of handwritten digits (28 x 28 px) into 10 categories (0-9)

## 01 - Movie Review Classification
Uses `binary_crossentropy` as the loss function.  The `relu` (rectified linar unit) is a function that zeroes out negative values while the last dense layer's output is based off of `sigmoid` which outputs between 0-1 (which is interpretted as a probability).

The purpose of the 20 EPOCH training run was so that there's sufficient data to graph the training/validation accuracy (and loss).  You can see that the graph diverts at around the 3rd EPOCH, which suggests overfitting.  Overfitting is when the training is memorizing the training data due to over-optimizization, hence it performs relatively poorly when given the validation set to validate against.  And this is why when i8t's rerun, it's done with only 4 EPOCHS, becuase that's all this approach needed to reach the peak that this model is going to produce (~86%).

## 02 - Newswire Multiclass Classification
Minor changes over binary classification -- most notably the use of `softmax` in the final layer, which is required to produce a probability of each of the classes (the sum of all the probabilities of all the classes should equal to 1.

There are two ways to handle labels multiclass classification:
- one-hot encode the labels + use `categorical_crossentropy` as the loss function
- encoding the labels as integers + use `sparse_categorical_crossentropy` as the loss function

If you need to classify large amount of categories, be mindful not to make your intermediate layers too small or you'll end up compressing the information into a space that is too low dimension which will decrease your model's accuracy significantly.

## 03 - House Price Regression
Dealing with prices of poses a problem: inflation.  You can't just keep feeding your neural network numbers without taking into consideration their ranges.  The solution to this is to do **feature-wise normalization**.

The neural network here will typically end with a single unit and no activation because it's predicting a scalar value and you don't want to put any restrictions as you would with a `sigmoid` or `softmax`.

*Mean squared error* (MSE) and *mean absolute error* (MAE) are also commonly used loss functions and metrics used in regression problems, respectively.

#### Feature-wise Normalization
For each feature/column in the input data:
- subtract mean of the feature
- divide by standard deviation
This will make it so that the feature is centered around 0 and has a *unit standard deviation*.  Here's an example of what that code might look like:
```
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
```
NOTE: even when normalizing test data, it should be computed using training data!

#### K-fold Cross-validation
Useed when you have few data points (i.e. small training dataset).  Normally (say you have a dataset with 10k entries), you would split it into a training and test (70:30 ratio), and with the training you'd divide that further into the actual training portion and a validation portion (i.e. out of the 7k training dataset, you might set aside 1k for validation use).  With a sufficiently large training dataset, this works well, but what if you have a small dataset (say, 500)?

When you have a small dataset, you're going to have a much higher variance in your validation scores as your sample size is smaller, so the way to handle this is use **K-fold cross-validation**.  What you do is you split your training dataset into K-paritions/folds (say, K=5), and then you would would do 5 runs on this data with a different partition actiing as the validation dataset each time, and then at the end your validation score would be the average validation score of your 5 runs.

You don't want to set it too high since you're going to be running your EPOCHS on for each fold (i.e. if K=10, EPOCHS=500, you're doing 5000 EPOCHS in total).  Typical K-values are 3-6.

**TL; DR** - K-fold cross-validation is a great way to reliably evaluate a model when you have little data

With k=5, I'm getting an avg validation MAE of ~2.5 and actual test MAE of ~2.75, and you might think that's pretty good/low, until you realize that the prices themselves are in the thousands (i.e. you're off by ~$2.75k) and the house prices back in the mid-1970's were in the 10k-50k range so being off by $2.75k is actually quite a bit... 


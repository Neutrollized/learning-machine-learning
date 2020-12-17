# README
The majority of the content in this folder are examples taken from [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) as [recommended reading](https://www.tensorflow.org/resources/learn-ml) from TensorFlow's site.  This book focuses on Deep Learning and has a lot of Keras examples; you may see a lot of overlap with content in the other folders as Keras is a popular library used in ML.

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


## 0x - Recognizing Handwritten Digits
The "Hello world" of deep learning with Keras.  Taken from the MNIST dataset that comes with Keras
- classify greyscale images of handwritten digits (28 x 28 px) into 10 categories (0-9)

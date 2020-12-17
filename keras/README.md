# README
The majority of the content in this folder are examples taken from [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) as [recommended reading](https://www.tensorflow.org/resources/learn-ml) from TensorFlow's site.  This book focuses on Deep Learning and has a lot of Keras examples; you may see a lot of overlap with content in the other folders as Keras is a popular library used in ML.

## Dependencies
- **keras**


## Data Representation in Neural Networks
Data is stored in multi-dimensional Numpy arrays called *tensors*.  It almost always contains numerical data.  Below are some common names and examples:

### Scalars (0D tensors)
```
import numpy as np
x = np.array(12)
print(x)
print(x.ndim)
```
Outputs:
```
12
0
```

### Vectors (1D tensors)
```
import numpy as np
x = np.array([12, 3, 6, 14])
print(x)
print(x.ndim)
```
Outputs:
```
[12 3 6 14]
1
```

### Matrices (2D tensors)
```
import numpy as np
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x)
print(x.ndim)
```
Outputs:
```
[[5 78 2 34 0]
 [6 79 3 35 1]
 [7 80 4 36 2]]
2
```

### 3D (and higher) tensors
```
import numpy as np
x = np.array([[[1, 78, 5, 34, 0],
               [2, 79, 6, 35, 1],
               [3, 80, 7, 36, 2]],
              [[4, 81, 8, 34, 0],
               [5, 82, 9, 35, 1],
               [6, 83, 0, 36, 2]],
              [[7, 84, 1, 34, 0],
               [8, 85, 2, 35, 1],
               [9, 86, 3, 36, 2]]])
print(x)
print(x.ndim)
```
Outputs:
```
[[[ 1 78  5 34  0]
  [ 2 79  6 35  1]
  [ 3 80  7 36  2]]

 [[ 4 81  8 34  0]
  [ 5 82  9 35  1]
  [ 6 83  0 36  2]]

 [[ 7 84  1 34  0]
  [ 8 85  2 35  1]
  [ 9 86  3 36  2]]]
3
```


## 0x - Recognizing Handwritten Digits
The "Hello world" of deep learning with Keras.  Taken from the MNIST dataset that comes with Keras
- classify greyscale images of handwritten digits (28 x 28 px) into 10 categories (0-9)

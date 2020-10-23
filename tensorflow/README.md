# README
If you're running these from a MacOS (as I am) or Linux, you will need to compile your own TensorFlow as the default package you install via pip currently won't have a lot of CPU extensions that will speed up your CPU-only training.  I have outlined some of the steps I took to compiling my own [**here**](https://gist.github.com/Neutrollized/6a409146fcf02438852c27633a031d0f)

## Dependencies
- **tensorflow** (it will install other package requirements, including **numpy**)
- **matplot**
- [**tensorflow-datasets**](https://blog.tensorflow.org/2019/02/introducing-tensorflow-datasets.html)
- [**seaborn**](https://seaborn.pydata.org/) (Auto MPG example)

`pip3 install tensorflow matplot tensorflow-datasets seaborn`


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

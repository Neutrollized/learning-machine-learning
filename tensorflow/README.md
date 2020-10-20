# README

## Dependencies
- **tensorflow** (it will install other package requirements, including **numpy**)
- **matplot**
- [**seaborn**](https://seaborn.pydata.org/) (Auto MPG example)

`pip3 install tensorflow matplot seaborn`


## 0 - Celcius to Fahrenheit (Regression)
- output: Number
- loss: Mean squared error (MSE)
- last layer activation fucntion: None


## 1 - Fashion MNIST (Classification)
- output: Probability distribution of its confidence of a certain class
- loss: Sparse categorical crossentropy
- last layer activation fucntion: Softmax

#### Exercises
Try the following settings to see the impact of it on the results/accuracy:
- set epochs = 1
- set number of neurons in Dense layer to range of something low (e.g. 10) to something high (e.g. 512)
- add additional dense layers
- don't normalize px values 


## 2 - [Auto MPG](https://www.youtube.com/watch?v=-vHQub0NXI4&ab_channel=TensorFlow)


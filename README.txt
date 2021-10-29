# Artificial Intelligence_A2

COMP 4190
Artificial Intelligence
Assignment 2

Authors:

Xiaojian Xie 7821950

YanLam Ng 7775665

Group: 9

Air Pollution Predictor:

Tensorflow version: v1

To run the program simply run the python code either for cleanData.py or multilayer_preceptron.py. 

How to run in terminal:

1.go the the Code directory 

2.run 'python {filename}'
    cleanData.py: filter, normalize and split the data.
    multilayer_preceptron.py: run multilayer_preceptron with cross validation

In the cleanData.py, we filter, normalize and split the data.

In the multilayer_preceptron.py, we use multilayer_preceptron to classify the Myo Keyboard data. We use cross validation to calculate the average accuracy.
To select the data for learning, we need to change the "learning_data" and "num_features" variable in the main method.
the learning_data variable determine which keyboard data and colomn is going to be learned. The To select the data for learning, we need to change the "learning_data" and "num_features" variable in the main method.
variable is the size of the keyboard data. 


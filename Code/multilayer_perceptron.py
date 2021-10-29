from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import statistics
from itertools import chain
# MNIST dataset parameters.

acce_column_names = ['x', 'y', 'z']
emg_column_names = ['emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']
gyro_column_names = ['x', 'y', 'z']
ori_column_names = ['x', 'y', 'z', 'w']
ori_euler_column_names = ['roll', 'pitch', 'yaw']

keys = ['Backward', 'Enter', 'Forward', 'Left', 'Right']
        #index:0    1           2           3       4
dataNames = [('accelerometer', acce_column_names),
             ('emg', emg_column_names),
             ('gyro', gyro_column_names),
             ('orientation', ori_column_names),
             ('orientationEuler', ori_euler_column_names)
             ]
weights = None
biases = None
optimizer = None
# Training parameters.
learning_rate = 0.002
training_steps = 2000
batch_size = 128
display_step = 100

# Network parameters.
n_hidden_1 = 518  # 1st layer number of neurons.
n_hidden_2 = 256  # 2nd layer number of neurons.
n_hidden_3 = 256

num_classes = 5
num_features = 2250  # data features ( 8*200+150+150+200+150)=2250.

def readDate(learning_data, test_index):
    test_y, test_x, train_y, train_x = [] ,[] ,[] ,[]

    for i in range(10):
        for key_index, key_name in enumerate(keys):
            temp_y, temp_x = [], []
            for dataName, col_list in dataNames:
                if dataName in learning_data:
                    fileName = '..\\Data\\output\\splitted\\{0}_{1}_{2}.csv'.format(key_name, dataName, i)
                    data = pd.read_csv(fileName)
                    df = pd.DataFrame(data, columns=col_list)

                    for col in col_list:
                        if col in learning_data[dataName]:
                            col_value = df[col].tolist()
                            temp_x.append(col_value )

            temp_x = list(chain.from_iterable(temp_x))

            if i == test_index:
                test_x.append(temp_x)
                test_y.append(key_index)
            else:
                train_x.append(temp_x)
                train_y.append(key_index)


    return (np.array(train_x), np.array(train_y)),(np.array(test_x), np.array(test_y))

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return norm

def cross_validate(learning_data, split_size=10):
  results = []
  global weights, biases,optimizer
  for i in range(split_size):
      # Prepare MNIST data.
      from tensorflow.keras.datasets import mnist
      (x_train, y_train), (x_test, y_test) = readDate(learning_data, i)

      # Convert to float32.
      x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
      # Flatten images to 1-D vector of 784 features (28*28).

      # Normalize images value from [0, 255] to [0, 1].


      # Use tf.data API to shuffle and batch data.
      train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
      train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

      # Store layers weight & bias

      # A random value generator to initialize weights.
      random_normal = tf.initializers.RandomNormal()


      weights = {
          'h1': tf.Variable(random_normal([num_features, n_hidden_1])),
          'h2': tf.Variable(random_normal([n_hidden_1, n_hidden_2])),
          'h3': tf.Variable(random_normal([n_hidden_2, n_hidden_3])),
          'out': tf.Variable(random_normal([n_hidden_3, num_classes]))
      }
      biases = {
          'b1': tf.Variable(tf.zeros([n_hidden_1])),
          'b2': tf.Variable(tf.zeros([n_hidden_2])),
          'b3': tf.Variable(tf.zeros([n_hidden_3])),
          'out': tf.Variable(tf.zeros([num_classes]))
      }
      # Stochastic gradient descent optimizer.
      optimizer = tf.optimizers.SGD(learning_rate)
      # Run training for the given number of steps.
      for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
          # Run the optimization to update W and b values.
          run_optimization(batch_x, batch_y)

          if step % display_step == 0:
              pred = neural_net(batch_x)
              loss = cross_entropy(pred, batch_y)
              acc = accuracy(pred, batch_y)
              print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

      # Test model on validation set.
      pred = neural_net(x_test)
      acc = accuracy(pred, y_test)
      print("Test Accuracy: %f" % acc)
      results.append(float(acc))


  print(results)
  return results



# Create model.
def neural_net(x):
    # Hidden fully connected layer with 128 neurons.
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Apply sigmoid to layer_1 output for non-linearity.
    layer_1 = tf.nn.sigmoid(layer_1)

    # Hidden fully connected layer with 256 neurons.
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Apply sigmoid to layer_2 output for non-linearity.
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # Output fully connected layer with a neuron for each class.
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(out_layer)



# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    global weights,biases,optimizer
    with tf.GradientTape() as g:
        pred = neural_net(x)
        loss = cross_entropy(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = list(weights.values()) + list(biases.values())

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))



if __name__ == '__main__':

    learning_data = {'accelerometer':['x', 'y', 'z'],
                     'emg': ['emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8'],
                     'gyro':['x', 'y', 'z'],
                     'orientation':['x', 'y', 'z', 'w'],
                     'orientationEuler':['roll', 'pitch', 'yaw']
                    }
    #features size of each emg column = 200
    #features size of each accelerometer, gyro, orientation, orientationEuler column = 50
    #all data features = 50*3 + 8*200 + 50*3 + 50*4 + 50*3 = 2250
    num_features = 2250
    result = cross_validate(learning_data)
    print('average accuracy:{0}'.format(statistics.mean(result)))

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import shuffle

import argparse
import sys

import tensorflow as tf

FLAGS = None

def readData(fname):
  parsedX = []
  parsedY = []
  with open(fname) as f:
    content = f.readlines()
  content = [x.strip() for x in content] 
  contentParsed = [text for text in content if len(text) > 0]
  
  for line in contentParsed:
    data = line.split(",")
    label = [1,0] if data[1] == "M" else [0,1]
    parsedY.append(label)
    parsedX.append([float(x) for x in data[2:]])
    
  f.close()
  
  return parsedX, parsedY

def customSoftmaxTrain(dataX, dataLabels, numClasses, validationPercent=0.2):
  assert len(dataX) == len(dataLabels)
  assert len(dataLabels) > 0
  assert len(dataLabels[0]) == numClasses
  
  dimensionWeight = len(dataX[0])

  allIndices = [i for i in range(len(dataX))]
  shuffle(allIndices)
  
  numValidation = int(validationPercent * len(dataX))
  
  validationX = [dataX[i] for i in allIndices[0:numValidation]]
  validationY = [dataLabels[i] for i in allIndices[0:numValidation]]
  
  trainX = [dataX[i] for i in allIndices[numValidation:]]
  trainY = [dataLabels[i] for i in allIndices[numValidation:]]
  
  def randomBatch(x, y, length):
    indices = [i for i in range(len(x))]
    shuffle(indices)
    indices = indices[0:length]
    xsubset = [x[i] for i in indices]
    ysubset = [y[i] for i in indices]
    return xsubset, ysubset
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, dimensionWeight])
  W = tf.Variable(tf.zeros([dimensionWeight, numClasses]))
  b = tf.Variable(tf.zeros([numClasses]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, numClasses])

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    #batch_xs, batch_ys = mnist.train.next_batch(20)
    batch_xs, batch_ys = randomBatch(trainX, trainY, 100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})        
    
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: validationX, y_: validationY}))  
  
  #batchX, batchY = randomBatch(dataX, dataLabels, 20)
  #feed_dict = {x: [batchX[0]]}
  #classification = y.eval(feed_dict)
  #print(classification)
  
  return x, y  
  
def main(_):
  dataX, dataY = readData("breast_cancer_data.txt")
  trainedX, trainedY = customSoftmaxTrain(dataX, dataY, 2)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

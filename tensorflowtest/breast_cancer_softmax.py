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
import numpy as np

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

# Not to make light of serious medical data, this is done to observe 
# if the mathematical can be extended and still be robust.
def generateData(originalData, originalLabels, numToGenerate, subsetSelection=2):
  assert subsetSelection >= 2
  assert subsetSelection <= len(originalData)
  assert len(originalData) == len(originalLabels)
  assert len(originalData) > 0
  assert numToGenerate > 0
  
  dimensionX = len(originalData[0])
  dimensionY = len(originalLabels[0])
  
  malignant = [originalData[i] for i in range(len(originalData)) if originalLabels[i][0] == 1]
  malignantLabels = [originalLabels[i] for i in range(len(originalData)) if originalLabels[i][0] == 1]
  benign = [originalData[i] for i in range(len(originalData)) if originalLabels[i][1] == 1]
  benignLabels = [originalLabels[i] for i in range(len(originalData)) if originalLabels[i][1] == 1]
  
  def randomBatch(x, y, length):
    indices = [i for i in range(len(x))]
    shuffle(indices)
    indices = indices[0:length]
    xsubset = [x[i] for i in indices]
    ysubset = [y[i] for i in indices]
    return xsubset, ysubset
  
  points, labels = [], []
  for _ in range(int(numToGenerate / 2.0)):
    batchX, batchY = randomBatch(malignant, malignantLabels, subsetSelection)
    
    avgPoint = [0 for _ in range(dimensionX)]
    for samplePoint in batchX:
      for i in range(dimensionX):
        avgPoint[i] += samplePoint[i]
    for i in range(dimensionX):
      avgPoint[i] /= len(batchX)

    points.append(avgPoint)
    labels.append([1,0])
    
    batchX, batchY = randomBatch(benign, benignLabels, subsetSelection)
    
    avgPoint = [0 for _ in range(dimensionX)]
    for samplePoint in batchX:
      for i in range(dimensionX):
        avgPoint[i] += samplePoint[i]
    for i in range(dimensionX):
      avgPoint[i] /= len(batchX)

    points.append(avgPoint)
    labels.append([0,1])
    
  return points, labels
  
def customSoftmaxTrain(dataX, dataLabels, numClasses, validationPercent, testX, testLabels):
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

  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.8, staircase=True)
  
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000): #1000
    #batch_xs, batch_ys = mnist.train.next_batch(20)
    batch_xs, batch_ys = randomBatch(trainX, trainY, 200) #200
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})        
    
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  false_positive = tf.logical_and( tf.not_equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.equal(tf.argmax(y, 1), 1) )
  false_negative = tf.logical_and( tf.not_equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.equal(tf.argmax(y, 1), 0) )
  #false_positive = tf.equal(True, false_positive)
  #false_negative = tf.equal(True, false_negative)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  train_acc = sess.run(accuracy, feed_dict={x: trainX, y_: trainY})
  valid_acc = sess.run(accuracy, feed_dict={x: validationX, y_: validationY})
  
  fp_result = sess.run(false_positive, feed_dict={x: validationX, y_: validationY})
  fn_result = sess.run(false_negative, feed_dict={x: validationX, y_: validationY})
  
  fp_percent, fn_percent = 0,0
  for dataBool in fp_result:
    if dataBool: fp_percent += 1
  for dataBool in fn_result:
    if dataBool: fn_percent += 1
  fp_percent /= len(fp_result)
  fn_percent /= len(fn_result)
  
  test_acc = sess.run(accuracy, feed_dict={x: testX, y_: testLabels})
  
  print("Training Accuracy: " + str(train_acc))
  print("Validation Accuracy: " + str(valid_acc))
  print("False Positive Rate (Validation): " + str(fp_percent))
  print("False Negative Rate (Validation): " + str(fn_percent))
  print("Loss: " + str(fn_percent * 3 + fp_percent))
  print("Speculative Test Accuracy: " + str(test_acc))
  print("------------------------")
  
  
  
  #batchX, batchY = randomBatch(dataX, dataLabels, 20)
  #feed_dict = {x: [batchX[0]]}
  #classification = y.eval(feed_dict)
  #print(classification)
  
  return x, y, train_acc, valid_acc, fp_percent, fn_percent
  
def main(_):
  dataX, dataY = readData("breast_cancer_data.txt")
  newPoints, newLabels = generateData(dataX, dataY, 500, 10)
  
  minLoss = 1.0
  bestX, bestY = None, None
  for _ in range(3):
    trainedX, trainedY, t_acc, v_acc, fp_percent, fn_percent = customSoftmaxTrain(dataX, dataY, 2, 0.2, newPoints, newLabels)
    loss = fn_percent * 3 + fp_percent
    if loss < minLoss:
      minLoss = loss
      bestX, bestY = trainedX, trainedY
  print(minLoss)
  
  
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

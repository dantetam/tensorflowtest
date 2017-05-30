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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import shuffle

import argparse
import sys

import tensorflow as tf
import numpy as np

from sklearn import datasets, linear_model

FLAGS = None

def quoteCustomSplit(text):
  #Find the true quotes in the message
  firstIndex, secondIndex = -1,-1
  for i in range(len(text)):
    c_i = text[i]
    c_l = text[i-1] if i > 0 else None
    c_r = text[i+1] if i < len(text) - 1 else None
    if c_i == '"' and c_l != "\\" and firstIndex == -1:
      firstIndex = i
    elif c_i == '"' and c_r == ',' and firstIndex != -1:
      secondIndex = i
  newText = text[0:firstIndex] + text[firstIndex:secondIndex].replace(",", "") + text[secondIndex:]
  return newText.split(",")
  
def readTwitterData(fname):
  parsedX = []
  parsedY = []
  with open(fname) as f:
    content = f.readlines()
  content = [x.strip() for x in content] 
  contentParsed = [text for text in content if len(text) > 0]
  
  for line in contentParsed:
    data = line.split(",")
    label = data[len(data) - 1]
    parsedY.append(label)
    
    newPoint = [float(x) for x in data[2:len(data) - 4]]
    
    if data[1] == "N": newPoint = [1,0,0,0,0] + newPoint
    if data[1] == "A": newPoint = [0,1,0,0,0] + newPoint
    if data[1] == "S": newPoint = [0,0,1,0,0] + newPoint
    if data[1] == "H": newPoint = [0,0,0,1,0] + newPoint
    if data[1] == "F": newPoint = [0,0,0,0,1] + newPoint
    
    if data[0] == "S": newPoint = [1,0,0,0] + newPoint
    if data[0] == "C": newPoint = [0,1,0,0] + newPoint
    if data[0] == "P": newPoint = [0,0,1,0] + newPoint
    if data[0] == "T": newPoint = [0,0,0,1] + newPoint
    
    parsedX.append(newPoint)
    
  f.close()
  
  return parsedX, parsedY
  
def linRegTrainTest(dataX, dataLabels, validationPercent, testX, testLabels):
  assert len(dataX) == len(dataLabels)
  assert len(dataLabels) > 0
  
  dimensionWeight = len(dataX[0])

  allIndices = [i for i in range(len(dataX))]
  shuffle(allIndices)
  
  numValidation = int(validationPercent * len(dataX))
  
  validationX = np.array([dataX[i] for i in allIndices[0:numValidation]])
  validationY = np.array([dataLabels[i] for i in allIndices[0:numValidation]])
  
  trainX = np.array([dataX[i] for i in allIndices[numValidation:]])
  trainY = np.array([dataLabels[i] for i in allIndices[numValidation:]])
  
  def randomBatch(x, y, length):
    indices = [i for i in range(len(x))]
    shuffle(indices)
    indices = indices[0:length]
    xsubset = [x[i] for i in indices]
    ysubset = [y[i] for i in indices]
    return xsubset, ysubset
    
  # Create linear regression object
  regr = linear_model.LinearRegression()

  # Train the model using the training sets
  regr.fit(trainX, trainY)

  # The coefficients
  print('Coefficients: \n', regr.coef_)
  # The mean squared error
  print("Mean squared error: %.2f"
        % np.mean((regr.predict(validationX) - validationY) ** 2))
  # Explained variance score: 1 is perfect prediction
  print('Variance score: %.2f' % regr.score(validationX, validationY))
  
def main(_):
  res = quoteCustomSplit('A,long,list,"of,things,to talk",continued')
  print(res)
  
  return
  
  dataX, dataY = readData("breast_cancer_data.txt")
  #dataX = tf.nn.l2_normalize(dataX, 0)
  newPoints, newLabels = generateData(dataX, dataY, 500, 10)
  
  minLoss = 1.0
  bestX, bestY = None, None
  for _ in range(3):
    trainedX, trainedY, t_acc, v_acc, fp_percent, fn_percent, trainedWeights = customSoftmaxTrain(dataX, dataY, 2, 0.2, newPoints, newLabels)
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

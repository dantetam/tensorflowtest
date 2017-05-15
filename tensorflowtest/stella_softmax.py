from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import shuffle

import argparse
import sys

# from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

from porter_stemmer import PorterStemmer

FLAGS = None

#TODO: Create a new data set for the sentences

def readInData(fname):
  with open(fname) as f:
    content = f.readlines()
  content = [x.strip() for x in content] 
  
  commandsById = dict()
  result = []
  workingOnCommand = False
  currentArr = []
  
  for text in content:
    if len(text) == 0 or '#' in text: #an empty line, which means a command has stopped
      workingOnCommand = False
      continue
    else:
      if not workingOnCommand: #beginning a new command
        if len(currentArr) > 0: #ignore leading spaces that don't actually signal ends of sentence groups
          result.append(currentArr)
        currentArr = []
        workingOnCommand = True
      else: #adding sentences to a certain command
        currentArr.append(text)
    
  if len(currentArr) > 0: #append the last set of sentences if not properly closed
    result.append(currentArr)
       
  f.close()     
       
  return result     
       
def convertSentencesToVector(arrSentences):
  stemmer = PorterStemmer()
  uniqueWords = set()
  
  stemmedSplit = [[] for _ in range(len(arrSentences))]
  
  for i in range(len(arrSentences)):
    arrSentence = arrSentences[i]
    for sentence in arrSentence:
      sentence = sentence.lower().strip(',.!?') #.replace(r"\(.*\)","")
      words = sentence.split()
      stemmedWords = [stemmer.stemword(word) for word in words]
      stemmedSplit[i].append(stemmedWords)
      for stemmedWord in stemmedWords:
        uniqueWords.add(stemmedWord)
        
  uniqueList = list(uniqueWords)
  uniqueWordsDict = {uniqueList[i]: i for i in range(len(uniqueList))}
  
  processedSentences = [[] for _ in range(len(stemmedSplit))]
  
  for i in range(len(stemmedSplit)):
    for j in range(len(stemmedSplit[i])):
      stemmedWords = stemmedSplit[i][j]
      processedSentences[i].append([])
      processedSentence = []
      for stemmedWord in stemmedWords:
        id = uniqueWordsDict[stemmedWord]
        processedSentences[i][j].append(id)
  
  return processedSentences, len(uniqueList)
   
def zeroOneEncode(processedSentences, size):
  results = [[] for _ in range(len(processedSentences))]
  for i in range(len(processedSentences)):
    for j in range(len(processedSentences[i])):
      processed = processedSentences[i][j]
      zeros = [0 for _ in range(size)]
      for id in processed:
        zeros[id] = 1
      results[i].append(zeros)
  return results
        
def customSoftmaxTrain(dataX, dataLabels, numClasses):
  assert len(dataX) == len(dataLabels)
  assert len(dataLabels) > 0
  assert len(dataLabels[0]) == numClasses
  
  dimensionWeight = len(dataX[0])

  def randomBatch(x, y, length):
    indices = [i for i in range(len(dataX))]
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
  for _ in range(10):
    #batch_xs, batch_ys = mnist.train.next_batch(20)
    batch_xs, batch_ys = randomBatch(dataX, dataLabels, 20)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})        
    
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: dataX, y_: dataLabels}))  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  #parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
  #FLAGS, unparsed = parser.parse_known_args()
  
  #tf.app.run(main=main, argv=[])
  #readInData("./commands_train.txt")
  #results, length = convertSentencesToVector([["This is a test", "just a test"],["a test of the outdoor warning system"]])
  #zeroOneEncode(results, length)
  
  fileSentences = readInData("./commands_train.txt")
  numClasses = len(fileSentences)
  results, length = convertSentencesToVector(fileSentences)
  zeroOneVector = zeroOneEncode(results, length)
  
  labels = []
  for i in range(len(results)):
    for _ in range(len(results[i])):
      label = [0 for _ in range(numClasses)]
      label[i] = 1
      labels.append(label)
    
  allZeroOneVectors = []
  for i in range(len(zeroOneVector)):
    for j in range(len(zeroOneVector[i])):
      vector = zeroOneVector[i][j]
      allZeroOneVectors.append(vector)
  
  customSoftmaxTrain(allZeroOneVectors, labels, numClasses)
  
  #print(allZeroOneVectors[27])
  #print(allZeroOneVectors[28])
  
  #print(labels[27])  
  #print(labels[28])  
  
  #print(numClasses)
  #print(zeroOneVector[0][0])
  #print(len(zeroOneVector[0][0]))
  
  
  
  
  
  
  
  
  
  
  

  # A comment to hold the line
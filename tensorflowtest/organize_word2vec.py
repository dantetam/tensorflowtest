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

#from sklearn import datasets, linear_model

import os, os.path

FLAGS = None

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
combos = [first + second for second in alphabet for first in alphabet]

def organizeWord2Vec(fname):
  with open(fname, encoding="latin-1") as f:
    for line in f:
      word = line.strip().split(" ")[0]
      for combo in combos:
        if word.lower().startswith(combo):
          with open('./word2vec/' + combo + ".txt",'a',encoding="latin-1") as f2:
            f2.write(line)
          break
      
def findVectorForWord(word):
  for combo in combos:
    if word.lower().startswith(combo):
      with open('./word2vec/' + combo + ".txt",'r',encoding="latin-1") as f:
        for line in f:
          tokens = line.strip().split(" ")
          if tokens[0] == word:
            return [np.float32(x) for x in tokens[1:]]
  return None    

vectorDim = 300  
def encodeSentences(sentences):
  #Determine the max length of the sentences, shorter sentences will be padded
  maxLen = 0
  for sentence in sentences:
    tokens = sentence.split()
    if len(tokens) > maxLen:
      maxLen = len(tokens)
  
  totalResult = np.ndarray(shape=(len(sentences),vectorDim*maxLen))
  
  for sentenceIndex in range(len(sentences)):
    sentence = sentences[sentenceIndex]
    vector = []
    tokens = sentence.split()
    for token in tokens:
      tokenVec = findVectorForWord(token)
      if tokenVec == None:
        #Initialize randomly as described in Convolutional Neural Networks, Yoon Kim
        tokenVec = [np.random.random_sample() for _ in range(vectorDim)]
      vector = vector + tokenVec
    padLen = maxLen - len(tokens)
    if padLen > 0:
      vector = vector + [0 for _ in range(padLen * 300)]
    totalResult[sentenceIndex] = vector
    
  return totalResult
  
def cnn_model_fn(features, labels, maxLenSentence, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, maxLenSentence, vectorDim, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, maxLenSentence, vectorDim, 1]
  # Output Tensor Shape: [batch_size, maxLenSentence, vectorDim, 8]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=8,
      kernel_size=[1, vectorDim],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, maxLenSentence, vectorDim, 8]
  # Output Tensor Shape: [batch_size, maxLenSentence, 1, 8]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, vectorDim], strides=vectorDim)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, maxLenSentence, 1, 8]
  # Output Tensor Shape: [batch_size, maxLenSentence * 1 * 8]
  pool1_flat = tf.reshape(pool2, [-1, maxLenSentence * 1 * 8])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, maxLenSentence * 1 * 8]
  # Output Tensor Shape: [batch_size, 8]
  dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=1-0.6, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 8]
  logits = tf.layers.dense(inputs=dropout, units=8)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)
      
def main(_):
  numFiles = len([name for name in os.listdir('./word2vec/')])
  if numFiles == 0:
    organizeWord2Vec("./word2vec_trained.txt")
  #print(findVectorForWord("Computer_Sciences"))
  sentences = ["This is a sentence", "another thing"]
  sentenceVecs = encodeSentences(sentences)
  labels = [1,0]
  print(sentenceVecs.shape)
  #cnn_model_fn(sentenceVecs, labels, 4, None)
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

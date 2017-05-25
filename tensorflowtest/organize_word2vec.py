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
#import numpy as np

#from sklearn import datasets, linear_model

import os, os.path

FLAGS = None

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
combos = [first + second for second in alphabet for first in alphabet]

def readWord2Vec(fname):
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
            return [float(x) for x in tokens[1:]]
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
      
def main(_):
  numFiles = len([name for name in os.listdir('./word2vec/')])
  if numFiles == 0:
    readWord2Vec("./word2vec_trained.txt")
  print(findVectorForWord("Computer_Sciences"))
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

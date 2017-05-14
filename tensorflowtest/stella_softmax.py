from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

from porter_stemmer import PorterStemmer

FLAGS = None

#TODO: Create a new data set for the sentences

def readInData(fname):
  stemmer = PorterStemmer()

  with open(fname) as f:
    content = f.readlines()
  content = [x.strip() for x in content] 
  
  commandsById = dict()
  result = []
  
  workingOnCommand = False
  
  currentArr = []
  
  for text in content:
    if len(text) == 0: #an empty line, which means a command has stopped
      workingOnCommand = False
      continue
    else:
      if not workingOnCommand: #beginning a new command
        if len(currentArr) > 0:
          result.append(currentArr)
        currentArr = []
        workingOnCommand = True
      else: #adding sentences to a certain command
        currentArr.append(sentence)
       
  return result     
       
        sentence = text.strip(',.!?').replace(r"\(.*\)","")
        words = sentence.split()
        stemmedWords = [stemmer.stemword(word) for word in words]
        currentArr.append(stemmedWords)
        
def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  
  testBatch = mnist.train.next_batch(1)
  print(testBatch[0])
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 26]))
  b = tf.Variable(tf.zeros([26]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 26])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  #print(sess.run(accuracy, feed_dict={x: mnist.test.images, #y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  #parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
  #FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[])
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf

import csv

COLUMNS = ["CATEGORY", "EMOTION", "TIME2_6", "TIME6_10", "TIME10_14",
           "TIME14_18", "TIME18_22", "TIME22_2", "DATE_SUN", "DATE_MON",
           "DATE_TUE", "DATE_WED", "DATE_THU", "DATE_FRI",
           "DATE_SAT", "PHOTO", "VIDEO", "ANIMATED_GIF", "LOG10_USER_FAV", "LOG10_USER_STATUS_COUNT", "TOPIC", "TEXT", "SANITIZED_TEXT", "SCORE"]
LABEL_COLUMN = "SCORE"
CATEGORICAL_COLUMNS = ["CATEGORY", "EMOTION"]
CONTINUOUS_COLUMNS = ["TIME2_6", "TIME6_10", "TIME10_14", "TIME14_18", "TIME18_22", "TIME22_2", 
            "DATE_SUN", "DATE_MON", "DATE_TUE", "DATE_WED", "DATE_THU", "DATE_FRI", "DATE_SAT", 
            "PHOTO", "VIDEO", "ANIMATED_GIF", "LOG10_USER_FAV", "LOG10_USER_STATUS_COUNT", 
            "TOPIC", "TEXT", "SANITIZED_TEXT"]

def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Sparse base columns.
  category = tf.contrib.layers.sparse_column_with_hash_bucket(
      "CATEGORY", hash_bucket_size=10)
  mood = tf.contrib.layers.sparse_column_with_hash_bucket(
      "MOOD", hash_bucket_size=10)

  # Continuous base columns.
  time1 = tf.contrib.layers.real_valued_column("TIME2_6")
  time2 = tf.contrib.layers.real_valued_column("TIME6_10")
  time3 = tf.contrib.layers.real_valued_column("TIME10_14")
  time4 = tf.contrib.layers.real_valued_column("TIME14_18")
  time5 = tf.contrib.layers.real_valued_column("TIME18_22")
  time6 = tf.contrib.layers.real_valued_column("TIME22_2")
  date1 = tf.contrib.layers.real_valued_column("DATE_SUN")
  date2 = tf.contrib.layers.real_valued_column("DATE_MON")
  date3 = tf.contrib.layers.real_valued_column("DATE_TUE")
  date4 = tf.contrib.layers.real_valued_column("DATE_WED")
  date5 = tf.contrib.layers.real_valued_column("DATE_THU")
  date6 = tf.contrib.layers.real_valued_column("DATE_FRI")
  date7 = tf.contrib.layers.real_valued_column("DATE_SAT")
  photo = tf.contrib.layers.real_valued_column("PHOTO")
  video = tf.contrib.layers.real_valued_column("VIDEO")
  gif = tf.contrib.layers.real_valued_column("ANIMATED_GIF")
  fav = tf.contrib.layers.real_valued_column("LOG10_USER_FAV")
  status = tf.contrib.layers.real_valued_column("LOG10_USER_STATUS_COUNT")
  
  # Transformations.
  # age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  wide_columns = [
      time1, time2, time3, time4, time5, time6,
      date1, date2, date3, date4, date5, date6, date7,
      photo, video, gif,
      fav, status]
                  
  deep_columns = [
      tf.contrib.layers.embedding_column(category, dimension=8),
      tf.contrib.layers.embedding_column(mood, dimension=8),
      time1, time2, time3, time4, time5, time6,
      date1, date2, date3, date4, date5, date6, date7,
      photo, video, gif,
      fav, status
  ]

  if model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        fix_global_step_increment_bug=True)
  return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  train_file_name, test_file_name = "short_tweets.csv", ""
  df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      engine="python",
      encoding='iso-8859-1',
      sep=',\s+',quoting=csv.QUOTE_ALL)
  """
  df_test = pd.read_csv(
      tf.gfile.Open(test_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1,
      engine="python")
  """

  # remove NaN elements
  #df_train = df_train.dropna(how='any', axis=0)
  #df_test = df_test.dropna(how='any', axis=0)

  df_train[LABEL_COLUMN] = (
      df_train["SCORE"].apply(lambda x: float(x) > 0.05)).astype(int)
  #df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir, model_type)
  m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
  #results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  #for key in sorted(results): print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=200,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

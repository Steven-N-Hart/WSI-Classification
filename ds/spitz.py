# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'spitz_%s-*.tfrecord'

SPLITS_TO_SIZES = {'train': 1000000, 'validation': 100000}

_NUM_CLASSES = 3

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 2',
}


def get_split(split_name, dataset_dir, file_pattern='spitz_train', reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """

  if not file_pattern:
    file_pattern = _FILE_PATTERN

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label': tf.FixedLenFeature(
          [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(shape=[299, 299,3]),
      'label': slim.tfexample_decoder.Tensor('image/class/label',shape=[]),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  #Count the total number of examples in all of these shard 
  num_samples = 0
  tfrecords_to_count = [os.path.join(dataset_dir, file) 
    for file in os.listdir(dataset_dir) 
    if file.startswith(file_pattern)]
  for tfrecord_file in tfrecords_to_count:
      for record in tf.python_io.tf_record_iterator(tfrecord_file):
          num_samples += 1
          
  return slim.dataset.Dataset(
      data_sources=tfrecords_to_count,
      reader=reader,
      decoder=decoder,
      num_samples=num_samples,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)

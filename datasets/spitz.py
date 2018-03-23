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

slim = tf.contrib.slim

_FILE_PATTERN = 'spitz'

_ITEMS_TO_DESCRIPTIONS = {
  'height': 'height of image',
  'width': 'width of image',
  'label': 'image label',
  'image_raw': 'raw image'
}


def get_split(split_name, dataset_dir, file_pattern='spitz', reader=None):
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
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  label_file = os.path.join(dataset_dir,'labels.txt')
  with tf.gfile.Open(label_file, 'r') as f:
    lines = f.read()
  lines = lines.split('\n')
  #return a dictionary of name (values) and index label(key)
  labels_to_names = { k:v for k,v in enumerate(lines)}

  #Count the total number of examples in all of these shards
  num_samples = 0
  tfrecords_to_count = [os.path.join(dataset_dir, file)
    for file in os.listdir(dataset_dir)
    if file.startswith(file_pattern+'_'+split_name)]

  for tfrecord_file in tfrecords_to_count:
    record_num=1
    print('\nExamining: {}'.format(tfrecord_file))
    try:
      for record in tf.python_io.tf_record_iterator(tfrecord_file):
          print('Record {}'.format(num_samples,record), end='\r', flush=True)
          num_samples += 1
          record_num += 1
    except:
        print('Died on record_num: {}.Regenerating file without the error'.format(record_num))

  return slim.dataset.Dataset(
      data_sources=tfrecords_to_count,
      reader=reader,
      decoder=decoder,
      num_samples=num_samples,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=len(labels_to_names),
      labels_to_names=labels_to_names)


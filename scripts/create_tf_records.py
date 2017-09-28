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
r"""
Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

tf.app.flags.DEFINE_string(
    'record_type', 'train', 'A value indicating either test or train [train].')

tf.app.flags.DEFINE_string(
    'output_directory', os.getcwd(), 'Where the output should go [cwd].')

tf.app.flags.DEFINE_string(
    'input_directory', None, 'Where the image directory input is.')

tf.app.flags.DEFINE_string(
    'prefix', 'train', 'What the prefix of tfrecords should be [train].')

tf.app.flags.DEFINE_integer(
    'num_shards', 5, 'How many shards should be created [5].')

tf.app.flags.DEFINE_integer(
    'random_seed', 1, 'Random seed [1].')

tf.app.flags.DEFINE_integer(
    'validation_count', 1000, 'Number of images to go in your validation set. [1000]')

FLAGS = tf.app.flags.FLAGS


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes():
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    directories = []
    class_names = []
    for filename in os.listdir(FLAGS.input_directory):
        path = os.path.join(FLAGS.input_directory, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id,prefix=FLAGS.prefix):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
        prefix, split_name, shard_id, FLAGS.num_shards)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(FLAGS.num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(FLAGS.num_shards):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) *
                                  num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                      sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                      sys.stdout.flush()
                      # Read the filename:
                      image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                      height, width = image_reader.read_image_dims(sess, image_data)
                      #print('Size is {} by {}'.format(height, width))
                      class_name = os.path.basename(os.path.dirname(filenames[i]))
                      class_id = class_names_to_ids[class_name]

                      example = dataset_utils.image_to_tfexample(
                        image_data, b'jpg', height, width, class_id)

                      tfrecord_writer.write(example.SerializeToString())
                      
    sys.stdout.write('\n')
    sys.stdout.flush()


def run(dataset_dir):

  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)


  photo_filenames, class_names = _get_filenames_and_classes()

  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(FLAGS.random_seed)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[FLAGS.validation_count:]
  validation_filenames = photo_filenames[:FLAGS.validation_count]
  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, class_names_to_ids,
                   dataset_dir)
  _convert_dataset('validation', validation_filenames, class_names_to_ids,
                   dataset_dir)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)


if __name__ == "__main__":
  run(FLAGS.output_directory)
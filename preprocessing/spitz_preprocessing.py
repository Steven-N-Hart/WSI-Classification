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
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

def preprocess_for_eval(image, height, width):
  image = tf.image.resize_images(image, [height, width]) 
  image = tf.image.per_image_standardization(image)
  return image

def preprocess_for_train(image, height, width):
  image = tf.image.resize_images(image, [height, width])
  image = tf.image.per_image_standardization(image)
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  return image

def preprocess_image(image, height, width,
                     is_training=False
                     ):
  """Pre-process one image for training or evaluation.

  Args:
    image: 3-D Tensor [height, width, channels] with the image. If dtype is
      tf.float32 then the range should be [0, 1], otherwise it would converted
      to tf.float32 assuming that the range is [0, MAX], where MAX is largest
      positive representable number for int(8/16/32) data type (see
      `tf.image.convert_image_dtype` for details).
    height: integer, image expected height.
    width: integer, image expected width.
    is_training: Boolean. If true it would transform an image for train,
      otherwise it would transform it for evaluation.

  Returns:
    3-D float Tensor containing an appropriately scaled image

  """
  #print('Preprcessing image',image,height,width)
  if is_training:
    return preprocess_for_train(image, height, width)
  else:
    return preprocess_for_eval(image, height, width)

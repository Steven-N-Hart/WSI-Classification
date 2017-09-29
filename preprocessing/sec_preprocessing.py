"""Secondary preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def preprocess_image(image):
  """Data argumentation. Produce child images from a same input image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.

  Returns:
    image_list - A secondary preprocessed images (an image list of child images).
    num_child_image - The number of child images.
    DA_flag - A flag that indicates this function has been called.
  """
# Set a flag indecating the function has been called.
#  DA_flag = 1

# Transform the image to floats.
  image = tf.to_float(image)
  
# Rotation
  # angle = [np.pi/4,np.pi/2,0.75*np.pi,np.pi,1.25*np.pi,1.5*np.pi,1.75*np.pi]    # rotate to 7 other orientations (4 ori in total)
  angle = [np.pi/2,np.pi,1.5*np.pi] 
  image_list = [image]    # initialize image_list; save orig image into image_list
  num_child_image = len(angle) + 1
  for ang in angle:
      image_single = tf.contrib.image.rotate(image,ang)
      image_list.append(image_single)
  return(image_list, num_child_image)
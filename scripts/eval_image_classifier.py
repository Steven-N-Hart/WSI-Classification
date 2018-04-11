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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import os

#os.environ["CUDA_VISIBLE_DEVICES"]=""

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_boolean('DA_flag', False, 'Data augmentation option. if True, then call "sec_preprocessing" func and average the logits.')

tf.app.flags.DEFINE_string(
    'sec_preprocessing_name', 'sec_preprocessing', 'The name of the secondary '
    'preprocessing to use. To create an image list. If left as `None`, then no '
    'image expansion will be used.')

FLAGS = tf.app.flags.FLAGS

def average_logits(logits, num_child_image, batch_size=FLAGS.batch_size):
    """ Calculate the average of multiple logits.  Useful when multiple image rotations used
    for a single prediction.
    """
    #print('Average Logits {}\n{}'.format(logits.shape,num_child_image ))
    logits_list = []
    n = 0
    while n < batch_size/num_child_image:
        logits_temp = tf.reduce_mean(logits[n*num_child_image:(n+1)*num_child_image], axis=0, keep_dims=True)
        logits_list.append(logits_temp)
        n = n + 1
    #logits = tf.concat(logits_list, 0)
    logits = tf.stack(logits_list)
    #print('Average Logits {}\t{}'.format(logits.shape,num_child_image ))
    return logits

def labels_average_logits(labels, num_child_image, batch_size=FLAGS.batch_size):
    """ When multiple image rotations used for a single prediction, they also have labels duplicated.
    This step essentially creates those labels.
    """
    #print('Average labels {}\n{}'.format(labels.shape,num_child_image ))
    label_list = []
    n = 0
    while n < batch_size/num_child_image:
        label_single = labels[n*num_child_image]
        label_list.append(label_single)
        n = n + 1
    #labels = tf.concat(label_list, 0)
    labels = tf.stack(label_list)
    labels = tf.reshape(labels,(tf.cast(batch_size/num_child_image,tf.int32),1))
    #print('Average labels {}\t{}\n{}'.format(labels.shape,num_child_image ))
    return labels



def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
    if FLAGS.DA_flag == True: #Data augmentation
        enqueue_many=True
        sec_preprocessing_name = FLAGS.sec_preprocessing_name
        image_sec_preprocessing_fn = preprocessing_factory.get_preprocessing(
            sec_preprocessing_name)
        image, num_child_image = image_sec_preprocessing_fn(image)
        #print('num_child_image: {}'.format(num_child_image))
        label = [label for i in range(num_child_image)]
        label=tf.reshape(label, [-1])
        #print('label: {}'.format(label))
        #print('image: {}'.format(image))
    else:
        enqueue_many=False

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    if FLAGS.DA_flag == True:
        # Need to reshape since now the shape is [batch_size, num_child_image, height, width, channels]
        # Needs to be [batch_size * num_child_image, height, width, channels]
        labels=tf.reshape(labels, [-1])
        labels=tf.reshape(labels, [FLAGS.batch_size * num_child_image, 1])

        images = tf.reshape(images,[-1])
        images=tf.reshape(images, [FLAGS.batch_size * num_child_image, eval_image_size,eval_image_size,3])

    ####################
    # Define the model #
    ####################
    logits, end_points = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)
    
    #print('Original predictions: {}'.format(predictions))
    #print('Original labels: {}'.format(labels))
    
    if FLAGS.DA_flag == True: #Actually score multiple image rotations of same image
        predictions = average_logits(predictions, num_child_image)
        AuxLogits = end_points['AuxLogits']
        AuxLogits = average_logits(AuxLogits, num_child_image)
        labels = labels_average_logits(labels, num_child_image)
        #print('predictions: {}'.format(predictions))
        #print('AuxLogits: {}'.format(AuxLogits))
        #print('labels: {}'.format(labels))
        #exit()



    #predictions = tf.Print(predictions,[predictions],"Predictions ",summarize=100)
    #predictions = tf.Print(predictions,[tf.confusion_matrix(labels,predictions),predictions.shape],'Differences',summarize=10)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'val/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        #'val/ROC': slim.metrics.streaming_curve_points(labels=labels, predictions=predictions)
        #'Recall_1': slim.metrics.streaming_recall_at_k(logits, labels, 1),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
    #slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        #checkpoint_dir=checkpoint_path,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()

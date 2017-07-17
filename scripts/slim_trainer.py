from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deployment import model_deploy
from ds import spitz as input_dataset
from nets import inception_v3

slim = tf.contrib.slim


#######################
# Hardware Flags #
#######################

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')


#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'spitz', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_train_name', 'train', 'The name of the train split.')

tf.app.flags.DEFINE_string(
    'dataset_validation_name', 'validation', 'The name of the validation split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/media/m087494/Padlock DT/Testing/TFRecords/', 
    'The directory where the TFRecords are stored.')

tf.app.flags.DEFINE_string(
    'out_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32,
    'How many images per batch')

tf.app.flags.DEFINE_integer(
    'num_epochs', None,
    'How many epocs to train on')

tf.app.flags.DEFINE_float(
    'learning_rate', 0.001,
    'HLearning rate')

tf.app.flags.DEFINE_integer(
    'image_size', 299,
    'Default image size to run through model')

FLAGS = tf.app.flags.FLAGS


#######################
# Helper methods      #
#######################


#######################
# Main methods      #
#######################
if not tf.gfile.Exists(FLAGS.out_dir):
  tf.gfile.MakeDirs(FLAGS.out_dir)

with tf.Graph().as_default():
  #######################
  # Config model_deploy #
  #######################
  deploy_config = model_deploy.DeploymentConfig(
      num_clones=FLAGS.num_clones,
      clone_on_cpu=FLAGS.clone_on_cpu,
      replica_id=FLAGS.task,
      num_replicas=FLAGS.worker_replicas,
      num_ps_tasks=FLAGS.num_ps_tasks
  )

  ##############################################################
  # Create a dataset provider that loads data from the dataset #
  ##############################################################
  with tf.device(deploy_config.inputs_device()):
    ds = input_dataset.get_split(
        split_name=FLAGS.dataset_name,
        dataset_dir=FLAGS.dataset_dir, 
        file_pattern='spitz_train')
    provider = slim.dataset_data_provider.DatasetDataProvider(ds,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10 * FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    print(image)
    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        allow_smaller_final_batch=True)
    # Define the model:
    print(images)
    logits, endpoints = inception_v3.inception_v3(images, is_training=True)
    print(labels)
    exit()
    # Specify the loss function:
    tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    train_tensor = slim.learning.create_train_op(total_loss, optimizer)

    # Actually runs training.
    slim.learning.train(train_tensor, FLAGS.out_dir)





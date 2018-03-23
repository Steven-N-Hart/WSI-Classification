#!/usr/bin/env python

from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
import os
import openslide
import logging

from colour import Color
from copy import copy
from PIL import Image


from nets import nets_factory
from preprocessing import preprocessing_factory


tf.app.flags.DEFINE_string('wsi',None, 'Whole slide image file')
tf.app.flags.DEFINE_string('label_file',None, 'Path to image label file that corresponds to TF graph.')
tf.app.flags.DEFINE_string('logLevel',None, 'Logging level to set [DEBUG, INFO, etc.]')

tf.app.flags.DEFINE_string('model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/from_scratch/inception_v3_rmsprop_0.01/','The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('eval_image_size', 299, 'Eval image size.')

tf.app.flags.DEFINE_string('output_prefix',None, 'Name of the output file prefix [default: `output`]')
tf.app.flags.DEFINE_integer('pixel_size', 10, 'Size to reduce image patch to (e.g. from 299 to 10)')
tf.app.flags.DEFINE_integer('level', 0, 'OpenSlide image level [default: `0`]')
tf.app.flags.DEFINE_integer('np_print', 210, 'Numpy Mean at which to run a patch [default: `210`]')
tf.app.flags.DEFINE_string('dict_file','summary.dict', 'Whether or not to append to dict.summary [default: `False`]')

FLAGS = tf.app.flags.FLAGS

######################################
# Do argument parsing and set defaults
######################################
if FLAGS.logLevel:
        logging.basicConfig(level=getattr(logging, FLAGS.logLevel))

tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


slim = tf.contrib.slim
model_name_to_variables = {'inception_v3':'InceptionV3','inception_v4':'InceptionV4','resnet_v1_50':'resnet_v1_50','resnet_v1_152':'resnet_v1_152'}

preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
eval_image_size = FLAGS.eval_image_size

model_variables = model_name_to_variables.get(FLAGS.model_name)
if model_variables is None:
  tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
  sys.exit(-1)

if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
  checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
else:
  checkpoint_path = FLAGS.checkpoint_path

MAX_SIZE = 8192 # Tesnsorflow can't handle bigger images




###########################
# Define helper functions #
###########################
def get_color_ranges(n):
  red = Color("red")
  blue = Color("blue")
  color_ranges = list(red.range_to(blue, n))
  rgb_ranges = [x.rgb for x in color_ranges]
  RGB_colors = [modify_rgb(x) for x in rgb_ranges]
  return(RGB_colors)

def modify_rgb(tup):
  RGB = []
  for x in tup:
    RGB.append(int(x*255))
  #RGBA.append(255) 
  return RGB

def get_labels(num_only=False):
  f = open(FLAGS.label_file, 'rb') 
  lines = f.readlines()

  labels = [str(w.decode("utf-8")).strip() for w in lines]
  logging.debug('Getting label information for {} labels'.format(len(labels)))
  """`label_colors` contains an array of RGBA colors that positionally
  correspond to the prediction scores
  """
  label_colors = get_color_ranges(len(labels))
  """ make a label dictionary so I can keep track of how many of each labels were scored """
  label_dict= {k:0 for (v,k) in enumerate(labels) }
  if num_only == True:
    return len(labels)+1
  return label_dict, label_colors, labels

def create_new_array(h_w_dim,RGBA_array):
  """ This function takes as input:

  h_w_dim: tuple of height, width, and dimensions
  RGBA_array: an array of values to fill the new array with
  
  returns single color array
  """
  #Create an array with dummy values
  beginning_array = np.arange(h_w_dim[0]*h_w_dim[1]*h_w_dim[2]).reshape(h_w_dim[0],h_w_dim[1],h_w_dim[2])
  beginning_array[:,:,:1] = RGBA_array[0]
  beginning_array[:,:,1:2] = RGBA_array[1]
  beginning_array[:,:,2:3] = RGBA_array[2]
  return beginning_array.astype(np.uint8)


def init_patch_constraints(img):
  """
  Determine what the patch size should be, and how many iterations it will take to get through the WSI
  The first level is always the main image
  Get width and height tuple for the first level
  """
  level = FLAGS.level
  img_dim = img.level_dimensions[level]
  np_mean_to_print = FLAGS.np_print
  patch_size = FLAGS.eval_image_size

  num_x_patches = int(img_dim[0] / patch_size) + 5
  num_y_patches = int(img_dim[1] / patch_size) + 5


  # find out how large a square I can make
  sprite_size = num_x_patches*FLAGS.pixel_size
  if sprite_size > MAX_SIZE:
    print('Your image plans are too large.  The max size is {} x {}, but you are requesting {} x {}'.format(
      MAX_SIZE, MAX_SIZE, sprite_size, sprite_size))
    exit()
  logging.debug('The WSI shape is {}'.format(img_dim))
  logging.debug('Making an output image of size {} x {}'.format(sprite_size,sprite_size))
  return num_x_patches,num_y_patches,patch_size


def init_output_images(num_x_patches,num_y_patches):
  #Initialize the scaled down image and heatmap
  master_im = Image.new(
    mode='RGBA',
    size=(num_x_patches*FLAGS.pixel_size,num_y_patches*FLAGS.pixel_size),
    color=None
    )

  master_hm = Image.new(
    mode='RGBA',
    size=(num_x_patches*FLAGS.pixel_size,num_y_patches*FLAGS.pixel_size),
    color=None
    )
  logging.debug('There are {} x-patches and {} y-patches to iterate through'.format(num_x_patches,num_y_patches))
  return master_im,master_hm

######################################
# Build tensor graph
######################################
image_string = tf.placeholder(tf.string) # Entry to the computational graph, e.g. image_string = tf.gfile.FastGFile(image_file).read()
image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True, acceptable_fraction=0.3) ## To process corrupted image files
image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)
network_fn = nets_factory.get_network_fn(FLAGS.model_name, get_labels(num_only=True), is_training=False)
if FLAGS.eval_image_size is None:
  eval_image_size = network_fn.default_image_size
processed_image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
processed_images  = tf.expand_dims(processed_image, 0)
logits, _ = network_fn(processed_images)
probabilities = tf.nn.softmax(logits)
# Initialize graph
init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))
sess = tf.Session()
init_fn(sess)



def parse_wsi():
  #Get labels and colors
  label_dict, label_colors, labels = get_labels()
  logging.debug('Found {} labels: {}'.format(len(label_dict),label_dict))

  # Open WSI
  img = openslide.OpenSlide(FLAGS.wsi)
  
  # Initialize patch info  
  num_x_patches,num_y_patches,patch_size = init_patch_constraints(img)
  number_of_useful_regions = 0

  # Initialize image
  master_im,master_hm = init_output_images(num_x_patches,num_y_patches)

  """ 
  This complicated for loop is designed to create a new heatmap image based on the WSI.

  If that region does not contain sufficient tissue, then its mean pixel value will be less than 
  `FLAGS.np_print`. In this case, we want to fill in this region with black.  Otherwise, we want
  to fill in pixels with the color that represents the most likely label.
  """
  for x in range(num_x_patches):
    
    # `x_top_left is` needed by openslide to know where to extract the region """
    # x=13
    x_top_left = x * patch_size
    if x % 10 == 0:
      logging.debug('Done with row {} of {} ({:.2f}%)'.format(x,num_x_patches,x/num_x_patches*100))

    for y in range(num_y_patches):
      # y=170
      y_top_left = y * num_y_patches
      # `y_top_left is` needed by openslide to know where to extract the region 
      """
      Now get the full image
      read_region(location, level, size)
      location (tuple): tuple giving the top left pixel
      level (int)     : the level number
      size (tuple)    : (width, height) tuple giving the region size
      """
      img_data = img.read_region((x_top_left,y_top_left),FLAGS.level, (patch_size, patch_size))
      img_data_np = np.array(img_data)
      
      im = Image.fromarray(img_data_np)
      image_data = img_data_np
      
      if np.mean(img_data_np) < FLAGS.np_print and np.mean(img_data_np) > 20: #Make sure the entire image isn't black
        """ If you've reached this spot, then there is something in the image.  So you should take that image, and 
        run it through the model to get a prediction
        """
        # logging.debug('Found a good image at x: {} and y: {} with a mean of {}'.
        # format(x_top_left,y_top_left,round(np.mean(img_data),2)))
        
        """ I am saving this to an image file, but only because I need to sort out how to get it in the right FastGFile format for TF 
        TODO: Make this not save to disk!
        """     
             
        ###########################################
        ### Start reading image and calling it ####
        im.save('img_data.jpg')
        image_data = tf.gfile.FastGFile('img_data.jpg', 'rb').read()
        image_contents =open('img_data.jpg', 'rb').read()
        predictions = sess.run(probabilities, feed_dict={image_string:image_contents})
        # logging.debug('Predictions: {}'.format(predictions))
        ### End reading image and calling it ####
        ###########################################
        
        # Calculate the difference between the best guesses
        pred_diff = np.diff(np.squeeze(predictions)[np.argsort(np.squeeze(predictions))][-2:])[0]
        # logging.debug('Pred diff: {}'.format(pred_diff))

        """Get the color corresponding to the prediction """
        section_color = label_colors[np.argmax(predictions)]
        # logging.debug('section_color : {}'.format(section_color))

        # Keep track of how many classifications I make of each type
        label_dict[labels[np.argmax(predictions)]] += 1
        number_of_useful_regions += 1
        print('\rlabel_dict : {}'.format(label_dict),end="")

        #Resze the image to the smaller format & Create a heatmap version
        if pred_diff > 0.1:
          image = im.resize( (FLAGS.pixel_size,FLAGS.pixel_size))
          master_im.paste(image,box=(x*FLAGS.pixel_size,y*FLAGS.pixel_size))
          heatmap = Image.fromarray(np.array(create_new_array((FLAGS.pixel_size,FLAGS.pixel_size,3),section_color)))
          master_hm.paste(heatmap,(x*FLAGS.pixel_size,y*FLAGS.pixel_size))
        else:
          # Not able to be scored. Append white.
          image = im.resize( (FLAGS.pixel_size,FLAGS.pixel_size))
          master_im.paste(image,box=(x*FLAGS.pixel_size,y*FLAGS.pixel_size))
          heatmap = Image.fromarray(np.array(create_new_array((FLAGS.pixel_size,FLAGS.pixel_size,3),(255,255,255))))
          master_hm.paste(heatmap,(x*FLAGS.pixel_size,y*FLAGS.pixel_size))


      else:
        # Not able to be scored.  Append white.
        image = im.resize( (FLAGS.pixel_size,FLAGS.pixel_size))
        master_im.paste(image,box=(x*FLAGS.pixel_size,y*FLAGS.pixel_size))
        heatmap = Image.fromarray(np.array(create_new_array((FLAGS.pixel_size,FLAGS.pixel_size,3),(255,255,255))))
        master_hm.paste(heatmap,box=(x*FLAGS.pixel_size,y*FLAGS.pixel_size))



  logging.info('number_of_useful_regions: {}'.format(number_of_useful_regions))
  master_im.save(str(FLAGS.output_prefix)+"_img.jpeg")

  #im = Image.fromarray(heatmap_array)
  master_hm.save(str(FLAGS.output_prefix)+"_heatmap.jpeg")
  os.remove('img_data.jpg')

  #Done with the image loop
  logging.info('Classification results: {}'.format(label_dict))
  data_count = []
  data_count.append(FLAGS.wsi)
  for key, value in label_dict.items():
    data_count.append(value)
  data_count = '\t'.join(str(x) for x in data_count)

  if not os.path.exists('summary.dict'):
    header= ['WSI']
    for key, value in label_dict.items():
      header.append(key)
    header = '\t'.join(header)
    with open('summary.dict','w') as f:
      f.write(header+"\n")

  with open('summary.dict','a') as f:
    f.write(data_count+"\n")


  sess.close()

if __name__ == '__main__':
  parse_wsi()










from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import openslide
import os
import pprint
import tensorflow as tf

from colour import Color
from copy import copy
from PIL import Image

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(description='Output a smaller image and heatmap')
parser.add_argument("-i", "--image", dest='wsi', required=True, help="path to a whole slide image")
parser.add_argument("-g", "--graph", dest='graph', required=True, help="path to tensorflow graph")
parser.add_argument("-l", "--labels", dest='label_file', required=True, help="path to image label file that corresponds to TF graph")
parser.add_argument("-o", "--output", dest='output_prefix', default="output", help="Name of the output file prefix [default: `output`]")
parser.add_argument("-s", "--size", dest='pixel_size', default=1, help="Size to reduce image patch to (e.g. from 299 to 10)")


parser.add_argument("-v", "--verbose",
            dest="logLevel",
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default="INFO",
            help="Set the logging level")

args = parser.parse_args()

pp = pprint.PrettyPrinter(indent=4)

if args.logLevel:
        logging.basicConfig(level=getattr(logging, args.logLevel))

level_0 = ''
Annotation_Number = 0


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

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



def create_new_array(h_w_dim,RGBA_array):
	""" This function takes as input:

	h_w_dim: tuple of height, width, and dimensions
	RGBA_array:	an array of values to fill the new array with
	
	returns single color array
	"""
	#Create an array with dummy values

	beginning_array = np.arange(h_w_dim[0]*h_w_dim[1]*h_w_dim[2]).reshape(h_w_dim[0],h_w_dim[1],h_w_dim[2])
	beginning_array[:,:,:1] = RGBA_array[0]
	beginning_array[:,:,1:2] = RGBA_array[1]
	beginning_array[:,:,2:3] = RGBA_array[2]
	return beginning_array.astype(np.uint8)

def parse_wsi():
	with tf.Session() as sess:
		""" Open up the graph """
		f = open(args.label_file, 'rb') 
		lines = f.readlines()

		labels = [str(w.decode("utf-8")).strip() for w in lines]
		logging.debug('Getting label information for {} labels'.format(len(labels)))
		"""`label_colors` contains an array of RGBA colors that positionally
		correspond to the prediction scores
		"""
		label_colors = get_color_ranges(len(labels))
		""" make a label dictionary so I can keep track of how many of each labels were scored """
		label_dict= {k:0 for (v,k) in enumerate(labels) }


		with tf.gfile.FastGFile(args.graph, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')
		
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

		"""
		The first level is always the main image
		Get width and height tuple for the first level
		"""
		img = openslide.OpenSlide(args.wsi)
		img_dim = img.level_dimensions[0]

		"""
		Determine what the patch size should be, and how many iterations it will take to get through the WSI
		"""
		level = 0
		np_mean_to_print = 210
		patch_size = 299
		patch_tup = (patch_size, patch_size)

		num_x_patches = int(img_dim[0] / patch_size) + 1
		num_y_patches = int(img_dim[1] / patch_size)+ 1

		#remainder_x = img_dim[1] % num_x_patches
		#remainder_y = img_dim[0] % num_y_patches

		logging.debug('The WSI shape is {}'.format(img_dim))
		logging.debug('There are {} x-patches and {} y-patches to iterate through'.format(num_x_patches,num_y_patches))

		""" This complicated for loop is designed to create a new heatmap image based on the WSI.

		If that region does not contain sufficient tissue, then its mean pixel value will be less than 
		`np_mean_to_print`. In this case, we want to fill in this region with black.  Otherwise, we want
		to fill in pixels with the color that represents the most likely label.

		"""
		new_image_array = []
		heatmap_array   = []
		
		number_of_useful_regions = 0

		for x in range(num_x_patches):
			"""`x_top_left is` needed by openslide to know where to extract the region """
			x_top_left = x * patch_size
			if x % 10 == 0:
					logging.debug('Done with row {} of {} ({:.2f}%)'.format(x,num_x_patches,x/num_x_patches*100))
			
			y_image_array = []
			y_heatmap_array = []

			for y in range(num_y_patches):
				"""`y_top_left is` needed by openslide to know where to extract the region """
				y_top_left = y * num_y_patches
				"""
				Now get the full image
				read_region(location, level, size)
				location (tuple): tuple giving the top left pixel
				level (int)     : the level number
				size (tuple)    : (width, height) tuple giving the region size
				"""
				img_data = img.read_region((x_top_left,y_top_left),level, (patch_size, patch_size))
				img_data_np = np.array(img_data)

				""" I am saving this to an image file, but only because I need to sort out how to get it in the right FastGFile format for TF 
				TODO: Make this not save to disk!
				"""

				im = Image.fromarray(img_data_np)
				image_data = img_data_np
				im.save('img_data.jpg')
				image_data = tf.gfile.FastGFile('img_data.jpg', 'rb').read()
				
				if np.mean(img_data_np) < np_mean_to_print:
					""" If you've reached this spot, then there is something in the image.  So you should take that image, and 
					run it through the model to get a prediction
					"""
					#logging.debug('Found a good image at x: {} and y: {} with a mean of {}'.
					#	format(x_top_left,y_top_left,round(np.mean(img_data),2)))
					
					predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
					#logging.debug('predictions: {}'.format(predictions))

					# exclude poor predictions
					pred_diff = np.diff(np.squeeze(predictions)[np.argsort(np.squeeze(predictions))][-2:])[0]

					
					
					"""Get the color corresponding to the prediction """
					section_color = label_colors[np.argmax(predictions)]


					number_of_useful_regions += 1
					label_dict[labels[np.argmax(predictions)]] += 1

					#Risze the image to the smaller format & Create a heatmap version
					if pred_diff > 0.1:
						image = np.array(im.resize( (args.pixel_size,args.pixel_size)))
						heatmap = np.array(create_new_array((args.pixel_size,args.pixel_size,3),section_color))
					else:
						# Not able to be scored.  Append white.
						image = np.array(im.resize( (args.pixel_size,args.pixel_size)))
						heatmap = np.array(create_new_array((args.pixel_size,args.pixel_size,3),(255,255,255)))

				else:
					# Not able to be scored.  Append white.
					image = np.array(im.resize( (args.pixel_size,args.pixel_size)))
					heatmap = np.array(create_new_array((args.pixel_size,args.pixel_size,3),(255,255,255)))

				if y == 0:
					y_image_array = image.flatten()
					y_heatmap_array = heatmap.flatten()
				else:
					y_image_array   = np.append(y_image_array,image.flatten())
					y_heatmap_array = np.append(y_heatmap_array,heatmap.flatten())
					

			# Completed row y
			y_image_array = y_image_array.reshape((
				args.pixel_size,
				num_y_patches*args.pixel_size,
				4))
			
			y_heatmap_array = y_heatmap_array.reshape((
				args.pixel_size,
				num_y_patches*args.pixel_size,
				3))

			#print(np.shape(y_image_array))
			#print(np.shape(y_heatmap_array))

			if x == 0:
				new_image_array = y_image_array
				heatmap_array   = y_heatmap_array
				#print('first new_image_array:{}'.format(new_image_array.shape))
			else:
				new_image_array = np.concatenate((new_image_array,y_image_array))
				heatmap_array = np.concatenate((heatmap_array,y_heatmap_array))

			# if x == 20:
			# 	logging.info('Classification results: {}'.format(label_dict))
			# 	logging.debug('number_of_useful_regions: {}'.format(number_of_useful_regions))
			# 	print('new_image_array:{}'.format(new_image_array.shape))
			# 	print('heatmap_array:{}'.format(heatmap_array.shape))
			# 	im = Image.fromarray(new_image_array)
			# 	im.save(str(args.output_prefix)+"_img.png")
			# 	im = Image.fromarray(heatmap_array)
			# 	im.save(str(args.output_prefix)+"_heatmap.png")
			# 	exit()


		logging.info('Classification results: {}'.format(label_dict))
		logging.debug('number_of_useful_regions: {}'.format(number_of_useful_regions))
		im = Image.fromarray(new_image_array)
		im.save(str(args.output_prefix)+"_img.jpeg")

		im = Image.fromarray(heatmap_array)
		im.save(str(args.output_prefix)+"_heatmap.jpeg")
		print('image_array:{}'.format(new_image_array.shape))
		os.remove('img_data.jpg')

if __name__ == '__main__':
	parse_wsi()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
try:
	import Image
except:
	from PIL import Image
import math
import numpy as np
import openslide
import os
from time import strftime,gmtime

parser = argparse.ArgumentParser(description='Extract a series of patches from a whole slide image')
parser.add_argument("-i", "--image", dest='wsi',  nargs='+', required=True, help="path to a whole slide image")
parser.add_argument("-p", "--patch_size", dest='patch_size', default=299, type=int, help="pixel width and height for patches")
parser.add_argument("-b", "--grey_limit", dest='grey_limit', default=0.8, type=float, help="greyscale value to determine if there is sufficient tissue present [default: `0.8`]")
parser.add_argument("-o", "--output", dest='output_name', default="output", help="Name of the output file directory [default: `output/`]")

parser.add_argument("-v", "--verbose",
			dest="logLevel",
			choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
			default="INFO",
			help="Set the logging level")
args = parser.parse_args()

if args.logLevel:
		logging.basicConfig(level=getattr(logging, args.logLevel))

wsi=' '.join(args.wsi)


""" Set global variables """
mean_grey_values = args.grey_limit * 255
number_of_useful_regions = 0
wsi=os.path.abspath(wsi)
outname=os.path.abspath(args.output_name)
basename = os.path.basename(wsi)
level = 0

def main():
	img,num_x_patches,num_y_patches = open_slide()
	logging.debug('img: {}, num_x_patches = {}, num_y_patches: {}'.format(img,num_x_patches,num_y_patches))
		
	for x in range(num_x_patches):
		for y in range(num_y_patches):
			img_data = img.read_region((x*args.patch_size,y*args.patch_size),level, (args.patch_size, args.patch_size))
			print_pics(x*args.patch_size,y*args.patch_size,img_data,img)

	pc_uninformative = number_of_useful_regions/(num_x_patches*num_y_patches)*100
	pc_uninformative = round(pc_uninformative,2)
	logging.info('Completed patch extraction of {} images.'.format(number_of_useful_regions))
	logging.info('{}% of the image is uninformative\n'.format(pc_uninformative))


def print_pics(x_top_left,y_top_left,img_data,img):
	if x_top_left % 100 == 0 and y_top_left % 100 == 0 and x_top_left != 0:
		pc_complete = round(x_top_left /img.level_dimensions[0][0],2) * 100
		logging.info('{:.2f}% Complete at {}'.format(pc_complete,strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
		exit()

	img_data_np = np.array(img_data)
	""" Convert to grayscale"""
	grey_img = rgb2gray(img_data_np)
	if np.mean(grey_img) < mean_grey_values:
		logging.debug('Image grayscale = {} compared to threshold {}'.format(np.mean(grey_img),mean_grey_values))
		global number_of_useful_regions
		number_of_useful_regions += 1
		img_name = os.path.basename(wsi) + "_" + str(x_top_left) + "_" + str(y_top_left) + "_" + str(args.patch_size)
		write_img_rotations(img_data_np,img_name)
		
def gen_x_and_y(xlist,ylist,img):
	for x in xlist:
		for y in ylist:
			img_data = img.read_region((x*args.patch_size,y*args.patch_size),level, (args.patch_size, args.patch_size))
			yield (x, y,img_data)

def open_slide():
	"""
	The first level is always the main image
	Get width and height tuple for the first level
	"""

	logging.debug('img: {}'.format(wsi))

	img = openslide.OpenSlide(wsi)
	img_dim = img.level_dimensions[0]

	"""
	Determine what the patch size should be, and how many iterations it will take to get through the WSI
	"""
	num_x_patches = int(math.floor(img_dim[0] / args.patch_size))
	num_y_patches = int(math.floor(img_dim[1] / args.patch_size))

	remainder_x = img_dim[0] % num_x_patches
	remainder_y = img_dim[1] % num_y_patches

	logging.debug('The WSI shape is {}'.format(img_dim))
	logging.debug('There are {} x-patches and {} y-patches to iterate through'.format(num_x_patches,num_y_patches))
	return img,num_x_patches,num_y_patches

def validate_dir_exists():
	if os.path.isdir(outname) == False:
		os.mkdir(outname)
	logging.debug('Validated {} directory exists'.format(outname))
	if os.path.exists(wsi):
		logging.debug('Found the file {}'.format(wsi))
	else:
		logging.debug('Could not find the file {}'.format(wsi))
		exit()

def rgb2gray(rgb):
	"""Converts an RGB image into grayscale """
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray

def save_image(img,j,img_name):
	try:
		im = Image.fromarray(img)
		im.save(os.path.join(outname,img_name+"_"+str(j)+".jpeg"))
	except:
		print('Could not print {}'.format(base_name))

def write_img_rotations(img,img_name):
	""" Have to print all these rotations since my preprocessing script
		will only accept 1 image at a time
	"""
	try:
		for i in range(4):
			save_image(img,i,img_name)
			img = np.rot90(img, k=1)
			
			img = np.fliplr(img)
			save_image(img,4,img_name)

		for i in range(5,9):
			save_image(img,i,img_name)
			img = np.rot90(img, k=1)		

		i+=1
		if i % 1000 == 0:
			print('{:.2f}% Complete with {} at {}'.format(i/num_images * 100,img_type,strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())),end="\r")

	except (RuntimeError, TypeError, NameError,IOError) as e:
		print('Failed to convert {}\n\t {}'.format(base_name,e))
	except:
		print('Died for another reason: {}'.format(base_name))
		exit()

if __name__ == '__main__':
	validate_dir_exists()
	main()
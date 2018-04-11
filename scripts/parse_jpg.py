from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2

import logging
try:
	import Image
except:
	from PIL import Image
import math
import numpy as np
import os


parser = argparse.ArgumentParser(description='Extract a series of patches from a jpeg image')
parser.add_argument("-i", "--image", dest='image_file',  required=True, help="path to an image")
parser.add_argument("-p", "--patch_size", dest='patch_size', default=299, type=int, help="pixel width and height for patches [default: `299`]")
parser.add_argument("-o", "--output", dest='output_name', default="output", help="Name of the output file directory [default: `output/`]")
parser.add_argument("-s", "--stride", dest='stride', default=0.5, type=float, help="Percent overlap for image patches [default: `0.5`]")

parser.add_argument("-v", "--verbose",
            dest="logLevel",
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default="INFO",
            help="Set the logging level")
args = parser.parse_args()

if args.logLevel:
        logging.basicConfig(level=getattr(logging, args.logLevel))



""" Set global variables """
image_path=os.path.abspath(r"args.image_file")
out_dir = os.path.abspath(args.output_name)

if not os.path.isdir(args.output_name):
	os.mkdir(args.output_name)

img = cv2.imread(image_path)
x_lim,y_lim,dims=img.shape

def get_file_name(image_file):
	return os.path.splitext(image_file)[0]

def print_image(file_name, x, y, img):
	fn = os.path.join(out_dir,os.path.basename(file_name)) + '_' + str(int(x)) + '_' + str(int(y)) + '.jpg'
	logging.debug('Writing to file: {}'.format(fn))
	logging.debug('Subsetting to {}:{},{}:{}'.format(x,x+args.patch_size,y,y+args.patch_size))
	subset_img =  img[x:x+args.patch_size, y:y+args.patch_size]
	if subset_img.shape == (args.patch_size,args.patch_size,3):
		cv2.imwrite(fn,subset_img)
	else:
		logging.debug('Not printing because of image shape:{}'.format(subset_img.shape))



file_name = get_file_name(args.image_file)
logging.debug('file: {}'.format(file_name))

adjusted_size = int(args.patch_size * args.stride)
x=0
y=0
while x < x_lim and x + args.patch_size < x_lim:
	while y < y_lim and y + args.patch_size < y_lim:
		logging.debug('X is now {} and Y is {}'.format(x,y))
		print_image(file_name,x, y, img)
		y = int(y + adjusted_size)
	y = 0
	x = int(x + adjusted_size)


#!/usr/bin/env python
"""
This script can be used to generate a sprite image from a direcotry of files
"""

import os
from PIL import Image
import glob
import argparse
from math import floor, sqrt
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", help="image directory")
parser.add_argument("--images_per_type", default=500, type=int,
					help="how many images to include for each type")
parser.add_argument("--subimage_size", default=50,type=int,
					help="number of pixels for each sub-image")
args = parser.parse_args()


image_dir = args.image_dir
images_per_type = args.images_per_type
subimage_size = args.subimage_size


MAX_SIZE = 8192  # tensorboard can't take images > this


# Get the subdirectories
sub_directories = [dI for dI in os.listdir(
	image_dir) if os.path.isdir(os.path.join(image_dir, dI))]

num_classes = len(sub_directories)

# Define the shape of the square
num_images_per_row = floor(sqrt(images_per_type * num_classes) )
print('planning {} images per row'.format(num_images_per_row))

# find out how large a square I can make
sprite_size = num_images_per_row*subimage_size
if sprite_size > MAX_SIZE:
	print('Your image plans are too large.  The max size is {} x {}, but you are requesting {} x {}'.format(
		MAX_SIZE, MAX_SIZE, sprite_size, sprite_size))
	exit()
print('Making an image of size {} x {}'.format(sprite_size,sprite_size))


# Initialize image
master = Image.new(
	mode='RGBA',
	size=(num_images_per_row*subimage_size,num_images_per_row*subimage_size),
	color=None
	)

print('Total size {}'.format(num_images_per_row*subimage_size))

# For each subdirectory, get a list of the first X files
count=defaultdict(int)
col_position = 0
row_position = 0

for i in range(num_classes):
	print('Printing {}'.format(sub_directories[i]))
	tmp_path = os.path.join(image_dir, sub_directories[i])
	images = os.listdir(tmp_path)[0:images_per_type]
	# make sure they match extension, so you don't get directories
	images = [img for img in images if img.endswith('jpeg')]

	# for each image, decrease the size
	for image_path in images:
		#print('Row position: {}\nCol_position: {}'.format(row_position,col_position))
		#Increment the count
		count[sub_directories[i]] += 1
		img = Image.open(os.path.join(tmp_path,image_path))
		img = img.resize((subimage_size, subimage_size), Image.NEAREST)
		#Paste into master
		master.paste(img,(col_position*subimage_size,row_position*subimage_size))
		if col_position + 1  > num_images_per_row:
			row_position += 1
			col_position = 0
		else:
			col_position +=1

master.save('master.jpeg')
with open('sprite_def.tsv','w') as f:
	f.write('Word\tFrequency\n')
	for k,v in count.items():
		f.write('{}\t{}\n'.format(k,v))

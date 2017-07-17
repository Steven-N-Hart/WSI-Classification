from __future__ import print_function
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import os
from time import gmtime, strftime

img_types=['Conventional','Other','Spitz']
img_types=['Other','Spitz']
base_image_dir='/projects/deepLearning/mayo/data/processed/Training-Images-Set-01/'
base_outdir = 'increased_images/'

def save_image(img,j,base_name):
    try:
        im = Image.fromarray(img)
        im.save(base_name[:-4]+"_"+str(j)+".jpeg")
    except:
        print('Could not print {}'.format(base_name))

for img_type in img_types:
    print('Making copies of {}'.format(img_type))
    image_dir=os.path.join(base_image_dir,img_type)
    outdir=os.path.join(base_outdir,img_type)
    i=0
    num_images=len(os.listdir(image_dir))
    for fname in os.listdir(image_dir):
        print('Analyzing {}'.format(fname),end="\r")
        try:
            img=mpimg.imread(os.path.join(image_dir,fname))
            base_name = os.path.basename(fname)
            base_name = os.path.join(outdir,base_name)
            save_image(img,0,base_name)
            img = np.rot90(img, k=1)
            save_image(img,1,base_name)
            img = np.rot90(img, k=1)
            save_image(img,2,base_name)
            img = np.rot90(img, k=1)
            save_image(img,3,base_name)
            img = np.fliplr(img)
            save_image(img,4,base_name)
            img = np.rot90(img, k=1)
            save_image(img,5,base_name)
            img = np.rot90(img, k=1)
            save_image(img,6,base_name)
            img = np.rot90(img, k=1)
            save_image(img,7,base_name)
            img = np.rot90(img, k=1)
            save_image(img,8,base_name)
            i+=1
            if i % 1000 == 0:
                print('{:.2f}% Complete with {} at {}'.format(i/num_images * 100,img_type,strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())),end="\r")    
        except (RuntimeError, TypeError, NameError,IOError) as e:
            print('Failed to convert {}\n\t {}'.format(base_name,e))
        except:
            print('Died for another reason: {}'.format(base_name))
            exit()


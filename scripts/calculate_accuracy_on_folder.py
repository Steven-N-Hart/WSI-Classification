import numpy as np
import tensorflow as tf
import sys
import glob


"""
This script takes a re-trained model and applys the classification to all images in a directory

calculate_accuracy_on_folder.py /path/toimage/dir

TODO: Add argparse and make all these paramters CL options
"""

imageDir = sys.argv[1]

modelFullPath = "/home/m087494/Tensorflow_project/m087494_Computer_Vision/models/model3/model3.pb"
labelsFullPath = "/home/m087494/Tensorflow_project/m087494_Computer_Vision/models/model3/output_labels.txt"


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    f = open(labelsFullPath, 'rb') 
    lines = f.readlines()
    labels = [str(w).strip() for w in lines]
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            for imagePath in glob.glob(imageDir+'/*/'+'*jpg'):
                image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
                predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
                predictions = np.ndarray.tolist(predictions)
                answer = []
                for i,pred in enumerate(predictions):
                   if i == 0:
                      answer.append(imagePath)
                   answer.append(pred)
                #answer.append(imagePath)
                #answer = ','.join([str(x) for x in answer])
                answer = ','.join(str(x) for x in answer)
                print(answer)

if __name__ == '__main__':
    run_inference_on_image()

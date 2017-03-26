import numpy as np
import tensorflow as tf
import os
import pandas as pd
from datetime import datetime

testPath = 'AAssignment_3_ML/test_images/'
modelFullPath = '/tmp/output_graph.pb'
labelsFullPath = '/tmp/output_labels.txt'

pred = []


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = None

    # Creates graph from the model stored in the tmp file once the training is complete.
    create_graph()

    count = 1
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        startTime = datetime.now()
        for i in range(1,6601):

            imagePath = testPath+'test_img-'+str(i)+'.jpg' #getting the actual path of the image    
            if not tf.gfile.Exists(testPath):
                tf.logging.fatal('File does not exist %s', imagePath)
                return answer

            image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

            predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-2:][::-1]  # Getting top 5 predictions
            f = open(labelsFullPath, 'rb')
            lines = f.readlines()
            labels = [str(w).replace("\n", "") for w in lines]

            answer = labels[top_k[0]]
            pred.append(answer)
            print count
            count = count + 1
        print datetime.now() - startTime


if __name__ == '__main__':
    run_inference_on_image()
    

    columns = ['class']
    pred = np.asarray(pred)
    df = pd.DataFrame(pred, columns=columns)
    df.index.name = 'id'
    df.to_csv("predictions_3.csv")

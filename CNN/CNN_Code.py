
# coding: utf-8

from __future__ import division, print_function, absolute_import

import numpy
from matplotlib import pyplot as plt
import tensorflow
import tflearn
from tflearn.data_utils import shuffle, to_categorical
import sklearn
from sklearn.cross_validation import train_test_split
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

trainX = numpy.load('tinyX.npy').transpose((0,3,2,1)).astype('float64') # this should have shape (26344, 3, 64, 64)
trainY = numpy.load('tinyY.npy') 
testX = numpy.load('tinyX_test.npy') # (6600, 3, 64, 64)


#Dataloading and preprocessing
trainX, trainY = shuffle(trainX, trainY)

X_train, X_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.20, random_state=42)
y_train = to_categorical(y_train, 40)
y_valid = to_categorical(y_valid, 40)

#
#Data preprocessing and augmentation
#


#Processing the images
print("Image Processing start")
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
print("Image Processing end")


#Augmenting the data
print("Image Aug start")
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
print("Image Aug end")



#
#Creating the neural network
#

#creating data model
cnn_network = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep, data_augmentation=img_aug)


#creating 2D Convolution
cnn_network = conv_2d(cnn_network, 64, 3, activation='relu')

#Adding a pooling layer
cnn_network = max_pool_2d(cnn_network, 2)

#Adding two 2D Convolution layers
cnn_network = conv_2d(cnn_network, 128, 3, activation='relu')
cnn_network = conv_2d(cnn_network, 128, 3, activation='relu')

#Adding a pooling layer
cnn_network = max_pool_2d(cnn_network, 2)

#Adding a fully connected layer
cnn_network = fully_connected(cnn_network, 512, activation='relu')

#Adding a 30% dropout layer
cnn_network = dropout(cnn_network, 0.3)

#Adding a fully connected layer with 40 outputs for the 40 categories
cnn_network = fully_connected(cnn_network, 40, activation='softmax')

#Specifying the hyper parameters and loss function used in the training
print("Specifying training parameters")
cnn_network = regression(cnn_network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

#training the network
print("Starting training")
model = tflearn.DNN(cnn_network, tensorboard_verbose=0)

print("start fitting")
model.fit(X_train, y_train, n_epoch=50, shuffle=True, validation_set=(X_valid, y_valid), show_metric=True, batch_size=96)
print("end fitting")

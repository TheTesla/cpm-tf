
# #### Run this on EC2 instance only
# 
# I am using tensorflow version 0.5.0 on my local Ubuntu machine. Way tooo old! 
# Need to upgrate it sometime soon.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

image_height = 480
image_width = 720

def load_data(test=False):
    fname = 'test.csv' if test else 'FLIC_dataset.csv'
    df = pd.read_csv(fname, nrows = 300)
    del df['Unnamed: 0']
    cols = df.columns[:-1]
    y = df[cols]
    
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0)
    df = df.dropna()
    X = np.vstack(df['Image'])
    X = X.reshape(-1, image_height, image_width, 1)
    
    if not test:
        y1 = y.ix[:,0:9] / 720.0
        y2 = y.ix[:,9:] / 480.0
        y = pd.concat([y1, y2], axis = 1)
        X, y = shuffle(X, y)

    else:
        y = None
    return X, y

x,y = load_data()
print (x.shape)

# To evaluate how good a prediction is
def eval_error(pred, ground_truth):
    return np.sqrt(mean_squared_error(pred, ground_truth))

from sklearn.cross_validation import train_test_split
x_train, x_valid, y_train,y_valid = train_test_split(x, y, test_size = 0.2, random_state=0)
print ("x_train has shape of:")
print (x_train.shape)


## -------------------------------------------------------------------------
## Here is the start of the TF graph
## -------------------------------------------------------------------------
batch_size = 32
image_height = 480
image_width = 720
num_channels = 1
num_labels = 18

deep_graph = tf.Graph()

with deep_graph.as_default():
    
    def conv2d(x, W, b, strides = 1):
        # tf.conv2d wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    
    def maxpool2d(x, k=2):
        # tf.max_pool wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def model_stage_1(x, weights, biases):
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        pool1 = maxpool2d(conv1, k=2)
        
        conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
        pool2 = maxpool2d(conv2, k=2)
        
        conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
        pool3 = maxpool2d(conv3, k=2)
        
        conv4 = conv2d(pool3, weights['wc4'], biases['bc4'])
        pool4 = maxpool2d(conv4, k=2)
        
        conv5 = conv2d(pool4, weights['wc5'], biases['bc5'])
        pool5 = maxpool2d(conv5, k=2)
        
        conv6 = conv2d(pool5, weights['wc6'], biases['bc6'])
        pool6 = maxpool2d(conv6, k=2)
        
        conv7 = conv2d(pool6, weights['wc7'], biases['bc7'])
        pool7 = maxpool2d(conv7, k=2)
        
        return tf.nn.relu(pool7)
        
    weights_stage_1 ={
        'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32])),
        'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64])),
        'wc3': tf.Variable(tf.truncated_normal([5, 5, 64, 128])),
        'wc4': tf.Variable(tf.truncated_normal([5, 5, 128, 256])),
        'wc5': tf.Variable(tf.truncated_normal([5, 5, 256, 512])),
        'wc6': tf.Variable(tf.truncated_normal([5, 5, 512, 1024])),
        'wc7': tf.Variable(tf.truncated_normal([5, 5, 1024, 2048]))
    }
    
    biases_stage_1 = {
        'bc1': tf.Variable(tf.zeros([32])),
        'bc2': tf.Variable(tf.zeros([64])),
        'bc3': tf.Variable(tf.zeros([128])),
        'bc4': tf.Variable(tf.zeros([256])),
        'bc5': tf.Variable(tf.zeros([512])),
        'bc6': tf.Variable(tf.zeros([1024])),
        'bc7': tf.Variable(tf.zeros([2048]))
    }
    
    def model_stage_2(x, weights, biases):
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        pool1 = maxpool2d(conv1, k=2)
        
        conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
        pool2 = maxpool2d(conv2, k=2)
        
        conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
        pool3 = maxpool2d(conv3, k=2)
        
        conv4 = conv2d(pool3, weights['wc4'], biases['bc4'])
        pool4 = maxpool2d(conv4, k=2)
        
        return tf.nn.relu(pool4)
        
    weights_stage_2 ={
        'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32])),
        'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64])),
        'wc3': tf.Variable(tf.truncated_normal([5, 5, 64, 128])),
        'wc4': tf.Variable(tf.truncated_normal([5, 5, 128, 256]))
    }
    
    biases_stage_2 = {
        'bc1': tf.Variable(tf.zeros([32])),
        'bc2': tf.Variable(tf.zeros([64])),
        'bc3': tf.Variable(tf.zeros([128])),
        'bc4': tf.Variable(tf.zeros([256]))
    }
    
    tf_train_dataset = tf.placeholder(
        tf.float32, 
        shape = (batch_size, image_height, image_width, num_channels))

    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))


    conv7_stage1 = model_stage_1(tf_train_dataset, weights_stage_1, biases_stage_1)
    print ("conv7_stage1 shape:")
    print (conv7_stage1.get_shape())
    
    conv4_stage2 = model_stage_2(tf_train_dataset, weights_stage_2, biases_stage_2)
    print ("conv4_stage2 shape")
    print (conv4_stage2.get_shape())
    
# Tried to figure out the pool_center_lower tensor but not working yet
#     pool_center_lower = tf.nn.avg_pool(tf_train_dataset, ksize=[1, 9, 9, 1], 
#                                        strides=[1, 9, 9, 1], padding='SAME')
#     print ("pool_center_lower shape: ")
#     print (pool_center_lower.get_shape())
    
tf.concat(0, [conv7_stage1, conv4_stage2])


## ------------------------------------------------------------------------
## Now we can use the TF graph
## ------------------------------------------------------------------------
print ("Start running TF graph ... ")
num_steps = 201

train_acc_records = np.zeros(num_steps)
valid_acc_records = np.zeros(num_steps)
loss_records = np.zeros(num_steps)

start_time = time.time()

with tf.Session(graph=deep_graph) as sess:
    #tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()
    print ("TF graph variables initialized ... ")
 
    for step in range(num_steps):
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)

        batch_data = x_train[offset:(offset + batch_size),:]
        batch_labels = y_train[offset:(offset + batch_size)]
        print (batch_labels.shape)
        
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels}

        _,l, pred = sess.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

        train_acc_records[step] = eval_error(pred, batch_labels)
        valid_acc_records[step] = eval_error(valid_prediction.eval(), y_valid)

        if (step % 20) == 0:
            print ("Minibatch loss at step %d: %f" % (step, l))
            print ("Minibatch RMSE: %0.5f" % train_acc_records[step])
            print ("Validation RMSE: %0.5f" % valid_acc_records[step])
            
    time_elasped = time.time() - start_time
    print ("==================================")
    print ("Net finished training!")
    print ("Run time is approx. %s minutes" % str(int(time_elapsed/60)))


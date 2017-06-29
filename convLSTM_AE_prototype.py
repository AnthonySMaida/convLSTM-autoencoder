#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:22:59 2017

@author: maida

This is a convolutional LSTM autoencoder prototype.
Has not been debugged.
Need to add code to create im4d.
"""

import sys
print("Python", sys.version)
import tensorflow as tf
print("TensorFlow: ", tf.__version__)

IM_SZ_LEN = 64 # For later experiments, increase size as necessary
IM_SZ_WID = 64
BATCH_SZ  = 1
NUM_UNROLLINGS = 3
LEARNING_RATE  = 0.1 # long story, may need simulated anealing
NUM_TRAINING_STEPS = 101

# Define im4d !!!
# Look at convolution 1x1 demo.

graph = tf.Graph()
with graph.as_default():
    
    # Use constant image for initial training
    # Look at grayscale conversion example to see how to read image
    x = tf.constant(im4d, dtype=tf.float32) # Define im4d !!!

    # Variable (wt) definitions. Only variables can be trained.
    # Naming conventions follow *Deep Learning*, Goodfellow et al, 2016.
    # input update
    U  = tf.Variable(tf.truncated_normal([5, 5, 2, 1], -0.1, 0.1))
    W  = tf.Variable(tf.truncated_normal([5, 5, 2, 1], -0.1, 0.1))
    B  = tf.Variable(tf.ones([IM_SZ_LEN, IM_SZ_WID]))

    # input gate (g_gate): input, prev output, bias
    Ug = tf.Variable(tf.truncated_normal([5, 5, 2, 1], -0.1, 0.1))
    Wg = tf.Variable(tf.truncated_normal([5, 5, 1, 1], -0.1, 0.1))
    Bg = tf.Variable(tf.ones([IM_SZ_LEN, IM_SZ_WID]))

    # forget gate (f_gate): input, prev output, bias
    Uf = tf.Variable(tf.truncated_normal([5, 5, 2, 1], -0.1, 0.1))
    Wf = tf.Variable(tf.truncated_normal([5, 5, 1, 1], -0.1, 0.1))
    Bf = tf.Variable(tf.ones([IM_SZ_LEN, IM_SZ_WID]))

    # output gate (q_gate): input, prev output, bias
    Uo = tf.Variable(tf.truncated_normal([5, 5, 2, 1], -0.1, 0.1))
    Wo = tf.Variable(tf.truncated_normal([5, 5, 1, 1], -0.1, 0.1))
    Bo = tf.Variable(tf.ones([IM_SZ_LEN, IM_SZ_WID]))

    initial_lstm_state  = tf.constant(tf.zeros([IM_SZ_LEN, IM_SZ_WID]))
    initial_lstm_output = tf.constant(tf.zeros([IM_SZ_LEN, IM_SZ_WID]))
    
    # The weights are global to this definition.
    def convLstmLayer(x, prev_s, prev_h):
        """ Build an convLSTM layer w/o peephole connections.
            Args x:      current input    (j elt row vec. batch_sz=1)
                 prev_h: previous output  (n elt row vec)
                 prev_s: previous state   (n elt row vec)
            Returns: h:     current ouput (n elt row vec)
                     state: current state (n elt row vec) """
        inp = tf.sigmoid(tf.nn.conv2d(x, U, [1, 1, 1, 1], padding='SAME')
                         + tf.nn.conv2d(prev_h, W, [1, 1, 1, 1], padding='SAME')
                         + B)
        g_gate = tf.sigmoid(tf.nn.conv2d(x, Ug, [1, 1, 1, 1], padding='SAME')
                            + tf.nn.conv2d(prev_h, Wg, [1, 1, 1, 1], padding='SAME')
                            + Bg)  # i_gate
        f_gate = tf.sigmoid(tf.nn.conv2d(x, Uf, [1, 1, 1, 1], padding='SAME')
                            + tf.nn.conv2d(prev_h, Wg, [1, 1, 1, 1], padding='SAME')
                            + Bf)
        q_gate = tf.sigmoid(tf.nn.conv2d(x, Uo, [1, 1, 1, 1], padding='SAME')
                            + tf.nn.conv2d(prev_h, Wo, [1, 1, 1, 1], padding='SAME')
                            + Bo)  # o_gate
        s = tf.multiply(f_gate, prev_s) + tf.multiply(g_gate, inp)
        h = tf.multiply(q_gate, tf.tanh(s)) # Also try logsig or relu
        return s, h

    # Since errorModule doesn't make reference to variables, it does not undergo training
    def errorModule(x, prediction):
        err1 = tf.nn.relu(x - prediction)
        err2 = tf.nn.relu(prediction - x)
        return tf.stack(err1, err2)
    
    # Build LSTM
    lstm_state  = initial_lstm_state
    lstm_output = initial_lstm_output
    for _ in range(NUM_UNROLLINGS): # three unrollings
        lstm_output, lstm_state = convLstmLayer(x, lstm_output, lstm_state)

    # "prediction" is always lstm_output
    error_module_output = errorModule(x, lstm_output)
    
    loss = tf.reduce_sum(error_module_output) # sums the values across each component of the tensor
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# Start training
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(NUM_TRAINING_STEPS): # 0 to 100
        _, l, predictions = session.run([optimizer, loss, lstm_output])
        
        print("Step: ", step)
        print("Loss: ", l)
        # print predicted image via some method

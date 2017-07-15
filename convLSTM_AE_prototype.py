#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:22:59 2017
Last modified: Wed July 6, 2017

@author: maida

This is a convolutional LSTM prototype for predictive coding.
It uses a constant image for training. 
Not yet debugged.
"""

import os
import sys
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import tensorflow as tf

print("Python version    :", sys.version)
print("TensorFlow version: ", tf.__version__)
print("Current directory : ", os.getcwd())

# For logging w/ TensorBoard
# The /tmp directory is periodically cleaned, such as on reboot.
# Since you probably don't want to keep these logs around forever,
# this is a practical place to put them.
LOGDIR = "/tmp/convLSTM/"

IM_SZ_LEN = 64 # For later experiments, increase size as necessary
IM_SZ_WID = 64
BATCH_SZ  = 1
NUM_UNROLLINGS = 2   # increase to 3 after debugging
#LEARNING_RATE  = 0.1 # long story, may need simulated anealing
NUM_TRAINING_STEPS = 1201

#model = tf.Graph()
#with model.as_default():
#    file_contents = tf.read_file('image_0004_leafCropped.jpg')
#    image         = tf.image.decode_jpeg(file_contents)
#    image         = tf.image.rgb_to_grayscale(image) # Input to the LSTM !!!
#    image         = tf.image.resize_images(image, [IM_SZ_LEN, IM_SZ_WID])
#    image         = tf.expand_dims(image, 0)
#    image         = (1/255.0) * image                # normalize to range 0-1
#    print("Shape of image: ", tf.shape(image))
#    print("Rank of  image: ", tf.rank(image))
#    print("Size of  image: ", tf.size(image))
#
#with tf.Session(graph=model) as sess:
#    print("Shape of image: ", tf.shape(image).eval())
#    print("Rank of  image: ", tf.rank(image).eval())
#    print("Size of  image: ", tf.size(image).eval())
#    output = sess.run(image)
#    
#
#print('Output shape after run() evaluation: ', output.shape)
#output.resize((IM_SZ_LEN, IM_SZ_WID))
#print('Resized for plt.imshow() :', output.shape)
#print('Print some matrix values to show it is grayscale.')
#print(output)
#print('Display the grayscale image.')
#plt.imshow(output, cmap = cm.Greys_r)
   
graph = tf.Graph()
with graph.as_default():
    
    file_contents = tf.read_file('image_0004_leafCropped.jpg')
    image         = tf.image.decode_jpeg(file_contents)
    image         = tf.image.rgb_to_grayscale(image) # Input to the LSTM !!!
    image         = tf.image.resize_images(image, [IM_SZ_LEN, IM_SZ_WID])
    image         = tf.expand_dims(image, 0)
    image         = (1/255.0) * image                # normalize to range 0-1

    # Variable (wt) definitions. Only variables can be trained.
    # Naming conventions follow *Deep Learning*, Goodfellow et al, 2016.
    # input update
    U  = tf.Variable(tf.truncated_normal([5, 5, 2, 1], -0.1, 0.1), name="U")
    W  = tf.Variable(tf.truncated_normal([5, 5, 1, 1], -0.1, 0.1), name="W")
    B  = tf.Variable(tf.ones([IM_SZ_LEN, IM_SZ_WID]),              name="B")
    B  = tf.expand_dims(B, 0)
    B  = tf.expand_dims(B, -1)

    # input gate (g_gate): input, prev output, bias
    Ug = tf.Variable(tf.truncated_normal([5, 5, 2, 1], -0.1, 0.1), name="Ug")
    Wg = tf.Variable(tf.truncated_normal([5, 5, 1, 1], -0.1, 0.1), name="Wg")
    Bg = tf.Variable(tf.ones([IM_SZ_LEN, IM_SZ_WID]),              name="Bg")
    Bg  = tf.expand_dims(Bg, 0)
    Bg  = tf.expand_dims(Bg, -1)

    # forget gate (f_gate): input, prev output, bias
    Uf = tf.Variable(tf.truncated_normal([5, 5, 2, 1], -0.1, 0.1), name="Uf")
    Wf = tf.Variable(tf.truncated_normal([5, 5, 1, 1], -0.1, 0.1), name="Wf")
    Bf = tf.Variable(tf.ones([IM_SZ_LEN, IM_SZ_WID]),              name="Bf")
    Bf  = tf.expand_dims(Bf, 0)
    Bf  = tf.expand_dims(Bf, -1)

    # output gate (q_gate): input, prev output, bias
    Uo = tf.Variable(tf.truncated_normal([5, 5, 2, 1], -0.1, 0.1), name="Uo")
    Wo = tf.Variable(tf.truncated_normal([5, 5, 1, 1], -0.1, 0.1), name="Wo")
    Bo = tf.Variable(tf.ones([IM_SZ_LEN, IM_SZ_WID]),              name="Bo")
    Bo  = tf.expand_dims(Bo, 0)
    Bo  = tf.expand_dims(Bo, -1)
  
    def newEmpty4Dtensor_1channel():
        """
           Returns a new 4D tensor with shape [1, 64, 64, 1].
           All elements are initialized to zero.
        """
        emptyTensor = tf.zeros([IM_SZ_LEN, IM_SZ_WID])
        emptyTensor = tf.expand_dims(emptyTensor, 0)
        emptyTensor = tf.expand_dims(emptyTensor, -1)
        return emptyTensor
    
    def newEmpty4Dtensor_2channels():
        """
           Returns a new 4D tensor with shape [1, 64, 64, 2].
           All elements are initialized to zero.
        """
        emptyTensor = tf.zeros([IM_SZ_LEN, IM_SZ_WID, 2])
        emptyTensor = tf.expand_dims(emptyTensor, 0)
        return emptyTensor
    
    # create some initializations
    initial_lstm_state  = newEmpty4Dtensor_1channel()
    initial_lstm_output = newEmpty4Dtensor_1channel()
    initial_err_input   = newEmpty4Dtensor_2channels()

    # The above weights are global to this definition.
    def convLstmLayer(err_inp, prev_s, prev_h):
        """ 
            Build an convLSTM layer w/o peephole connections.
            Input args: 
                 err_inp:  current input    (tensor: [1, 64, 64, 2])
                 prev_h :  previous output  (tensor: [1, 64, 64, 1])
                 prev_s :  previous state   (tensor: [1, 64, 64, 1])
            Returns: 
                     s  :  current state    (tensor: [1, 64, 64, 1])
                     h  :  current output   (tensor: [1, 64, 64, 1])
        """
        with tf.name_scope("LSTM"):
          inp = tf.sigmoid(tf.nn.conv2d(err_inp, U, [1, 1, 1, 1], padding='SAME')
                         + tf.nn.conv2d(prev_h, W, [1, 1, 1, 1], padding='SAME')
                         + B, name="inp")
          g_gate = tf.sigmoid(tf.nn.conv2d(err_inp, Ug, [1, 1, 1, 1], padding='SAME')
                            + tf.nn.conv2d(prev_h, Wg, [1, 1, 1, 1], padding='SAME')
                            + Bg, name="g_gate")  # i_gate is more common name
          f_gate = tf.sigmoid(tf.nn.conv2d(err_inp, Uf, [1, 1, 1, 1], padding='SAME')
                            + tf.nn.conv2d(prev_h, Wg, [1, 1, 1, 1], padding='SAME')
                            + Bf, name="f_gate")
          q_gate = tf.sigmoid(tf.nn.conv2d(err_inp, Uo, [1, 1, 1, 1], padding='SAME')
                            + tf.nn.conv2d(prev_h, Wo, [1, 1, 1, 1], padding='SAME')
                            + Bo, name="q_gate")  # o_gate is more common name
          s = tf.add(tf.multiply(f_gate, prev_s), tf.multiply(g_gate, inp), name="state")
          h = tf.multiply(q_gate, tf.sigmoid(s), name="output") # Also try relu
          return s, h       # normally above is tanh

    # errorModule doesn't use variables, so doesn't undergo training
    def errorModule(image, predict):
        """
            Build an error representation for input to the convLSTM layer.
            Input args:
                image:   target image         (tensor: [1, 64, 64, 1])
                predict: predicted image      (tensor: [1, 64, 64, 1])
            Returns:
                tensor4D: Errs packed in 2 channels. (tensor: [1, 64, 64, 2])
        """
        with tf.name_scope("ErrMod"):
          err1     = tf.nn.relu(image - predict, name="E1")
          err2     = tf.nn.relu(predict - image, name="E2")
          tensor5D = tf.stack([err1, err2], axis=3)
          tensor4D = tf.reshape(tensor5D, [1, IM_SZ_LEN, IM_SZ_WID, 2], name="PrdErr")
          return tensor4D
    
    # Build LSTM
    lstm_state  = initial_lstm_state
    lstm_output = initial_lstm_output
    err_input   = initial_err_input
    with tf.name_scope("full_model"):
        for _ in range(NUM_UNROLLINGS): # three unrollings
            lstm_state, lstm_output = convLstmLayer(err_input, lstm_state, lstm_output)
            err_input               = errorModule(image, lstm_output)

    # "prediction" is always lstm_output
#    error_module_output = errorModule(x, lstm_output)

    #New optimizer block, uses exp decay on learning rate, added clip_by_global_norm
    loss = tf.reduce_sum(err_input) # sums the values across each component of the tensor
    global_step = tf.Variable(0)

    #learning rate starts at 10, decreases by 90% every 300 steps
    learning_rate  = tf.train.exponential_decay(
        10.0, global_step, 300, 0.1, staircase=True, name='LearningRate')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients,1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients,v),global_step=global_step)
    
    with tf.name_scope("initializations"):
        tf.summary.image("initial_lstm_state", initial_lstm_state, 3)
        tf.summary.image("initial_lstm_output", initial_lstm_output, 3)
        tf.summary.image("initial_error1", 
                         tf.slice(initial_err_input, [0,0,0,0], [1, 64, 64, 1]), 3)
        tf.summary.image("initial_error2",
                         tf.slice(initial_err_input, [0,0,0,1], [1, 64, 64, 1]), 3)
    with tf.name_scope("input"):
        tf.summary.image("image", image, 3)
    with tf.name_scope("lstm"):
        tf.summary.image("lstm_out", lstm_output, 3)
        tf.summary.image("lstm_state", lstm_state, 3)
    with tf.name_scope("error"):
        tf.summary.image("perror_1", 
                         tf.slice(err_input, [0,0,0,0], [1, 64, 64, 1]), 3)
        tf.summary.image("perror_2", 
                         tf.slice(err_input, [0,0,0,1], [1, 64, 64, 1]), 3)
    with tf.name_scope('optimizer'):
        tf.summary.scalar('loss',loss)
        tf.summary.scalar('learning_rate',learning_rate)

# Start training
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    
    # Create graph summary
    # Use a different log file each time you run the program.
    msumm = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR + "1") # += 1 for each run till /tmp is cleard
    writer.add_graph(sess.graph)

    print("Shape of image: ", tf.shape(image).eval())
    print("Rank of  image: ", tf.rank(image).eval())
    print("Size of  image: ", tf.size(image).eval())
    
    print("Shape of initial_lstm_state: ", tf.shape(initial_lstm_state).eval())
    print("Rank of  initial_lstm_state: ", tf.rank(initial_lstm_state).eval())
    print("Size of  initial_lstm_state: ", tf.size(initial_lstm_state).eval())

    print("Shape of lstm_state: ", tf.shape(lstm_state).eval())
    print("Rank of  lstm_state: ", tf.rank(lstm_state).eval())
    print("Size of  lstm_state: ", tf.size(lstm_state).eval())

    print("Shape of initial_lstm_output: ", tf.shape(initial_lstm_output).eval())
    print("Rank of  initial_lstm_output: ", tf.rank(initial_lstm_output).eval())
    print("Size of  initial_lstm_output: ", tf.size(initial_lstm_output).eval())

    print("Shape of lstm_output: ", tf.shape(lstm_output).eval())
    print("Rank of  lstm_output: ", tf.rank(lstm_output).eval())
    print("Size of  lstm_output: ", tf.size(lstm_output).eval())

    print("Shape of initial_err_input: ", tf.shape(initial_err_input).eval())
    print("Rank of  initial_err_input: ", tf.rank(initial_err_input).eval())
    print("Size of  initial_err_input: ", tf.size(initial_err_input).eval())

    print("Shape of err_input: ", tf.shape(err_input).eval())
    print("Rank of  err_input: ", tf.rank(err_input).eval())
    print("Size of  err_input: ", tf.size(err_input).eval())

# Below would only used to test if the input makes sense
#    output = sess.run(image)

    for step in range(NUM_TRAINING_STEPS): # 0 to 100
        if step % 1 == 0:
            ms = sess.run(msumm)
            writer.add_summary(ms, step)
        _, l, predictions = sess.run([optimizer, loss, lstm_output])
        
        print("Step: ", step)
        print("Loss: ", l)








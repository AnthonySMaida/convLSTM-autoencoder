#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:22:59 2017
Last modified: Sept 3, 2017

@author: maida, kirby

This is a convolutional LSTM prototype for predictive coding.
It uses a constant image for training. 
"""

import os
import sys
import numpy as np
import matplotlib as mpl
from matplotlib import cm, colors
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import tensorflow as tf

print("Python version    : ", sys.version)
print("TensorFlow version: ", tf.VERSION)
print("Current directory : ", os.getcwd())
print("Matplotlib version: ", mpl.__version__)

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
NUM_TRAINING_STEPS = 11
#NUM_TRAINING_STEPS = 1201
   
graph = tf.Graph()
with graph.as_default():
    
    file_contents = tf.read_file('image_0004_leafCropped.jpg')
    image         = tf.image.decode_jpeg(file_contents)
    image         = tf.image.rgb_to_grayscale(image) # Input to Error mod !!!
    image         = tf.image.resize_images(image, [IM_SZ_LEN, IM_SZ_WID])
    image         = tf.expand_dims(image, 0) # inserts dimension of 1 at front
                                             # of tensor
    # input to error module
    image         = (1/255.0) * image                # normalize to range 0-1

    # Variable (wt) definitions. Only variables can be trained.
    # Naming conventions follow *Deep Learning*, Goodfellow et al, 2016.
    # Using seeds to get repeatable results FOR DEBUGGING ONLY.
    
    # U wts have two channels to match output of error module.

    # input update
    with tf.name_scope('Input_Update_Weights'):
        U  = tf.Variable(tf.truncated_normal([5, 5, 2, 1], mean=-0.1, stddev=0.1, seed=1), name="U")
        W  = tf.Variable(tf.truncated_normal([5, 5, 1, 1], mean=-0.1, stddev=0.1, seed=2), name="W")
        B  = tf.Variable(tf.ones([1, IM_SZ_LEN, IM_SZ_WID,1 ]),                            name="B")

    # input gate (g_gate): input, prev output, bias
    with tf.name_scope('Input_Gate_Weights'):
        Ug = tf.Variable(tf.truncated_normal([5, 5, 2, 1], mean=-0.1, stddev=0.1, seed=3), name="Ug")
        Wg = tf.Variable(tf.truncated_normal([5, 5, 1, 1], mean=-0.1, stddev=0.1, seed=4), name="Wg")
        Bg = tf.Variable(tf.ones([1, IM_SZ_LEN, IM_SZ_WID,1 ]),                            name="Bg")

    # forget gate (f_gate): input, prev output, bias
    with tf.name_scope('Forget_Gate_Weights'):
        Uf = tf.Variable(tf.truncated_normal([5, 5, 2, 1], mean=-0.1, stddev=0.1, seed=5), name="Uf")
        Wf = tf.Variable(tf.truncated_normal([5, 5, 1, 1], mean=-0.1, stddev=0.1, seed=6), name="Wf")
        Bf = tf.Variable(tf.ones([1, IM_SZ_LEN, IM_SZ_WID, 1]),                            name="Bf")

    # output gate (q_gate): input, prev output, bias
    with tf.name_scope('Output_Gate_Weights'):
        Uo = tf.Variable(tf.truncated_normal([5, 5, 2, 1], mean=-0.1, stddev=0.1, seed=7), name="Uo")
        Wo = tf.Variable(tf.truncated_normal([5, 5, 1, 1], mean=-0.1, stddev=0.1, seed=8), name="Wo")
        Bo = tf.Variable(tf.ones([1, IM_SZ_LEN, IM_SZ_WID, 1]),                            name="Bo")
  
    def newEmpty4Dtensor_1channel():
        """
           Returns a new 4D tensor with shape [1, 64, 64, 1].
           All elements are initialized to zero.
        """
        emptyTensor = tf.zeros([1, IM_SZ_LEN, IM_SZ_WID, 1])
        return emptyTensor
    
    def newEmpty4Dtensor_2channels():
        """
           Returns a new 4D tensor with shape [1, 64, 64, 2].
           All elements are initialized to zero.
        """
        emptyTensor = tf.zeros([1, IM_SZ_LEN, IM_SZ_WID, 2])
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
                            + tf.nn.conv2d(prev_h, Wf, [1, 1, 1, 1], padding='SAME')
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

    with tf.name_scope('weights'):
        with tf.name_scope('1_input_update'):
            newU1 = tf.slice(U,[0,0,0,0],[5,5,1,1])
            newU2 = tf.slice(U,[0,0,1,0],[5,5,1,1])
            newW = tf.slice(W,[0,0,0,0],[5,5,1,1])
            newU1 = tf.squeeze(newU1)     #now a viewable [5x5] matrix
            newU2 = tf.squeeze(newU2)
            newW = tf.squeeze(newW)
            newU1 = tf.reshape(newU1,[1,5,5,1])
            newU2 = tf.reshape(newU2,[1,5,5,1])
            newW = tf.reshape(newW,[1,5,5,1])
            tf.summary.image('U1', newU1)
            tf.summary.image('U2', newU2)
            tf.summary.image('W', newW)
            tf.summary.image('B', B)
            
        with tf.name_scope('2_input_gate'):
            newUg1 = tf.slice(Ug,[0,0,0,0],[5,5,1,1])
            newUg2 = tf.slice(Ug,[0,0,1,0],[5,5,1,1])
            newWg = tf.slice(Wg,[0,0,0,0],[5,5,1,1])
            newUg1 = tf.squeeze(newUg1)     #now a viewable [5x5] matrix
            newUg2 = tf.squeeze(newUg2)
            newWg = tf.squeeze(newWg)
            newUg1 = tf.reshape(newUg1,[1,5,5,1])
            newUg2 = tf.reshape(newUg2,[1,5,5,1])
            newWg = tf.reshape(newWg,[1,5,5,1])
            tf.summary.image('Ug1', newUg1)
            tf.summary.image('Ug2', newUg2)
            tf.summary.image('Wg', newWg)
            tf.summary.image('Bg', Bg)

        with tf.name_scope('3_forget_gate'):
            newUf1 = tf.slice(Uf,[0,0,0,0],[5,5,1,1])
            newUf2 = tf.slice(Uf,[0,0,1,0],[5,5,1,1])
            newWf = tf.slice(Wf,[0,0,0,0],[5,5,1,1])
            newUf1 = tf.squeeze(newUf1)     #now a viewable [5x5] matrix
            newUf2 = tf.squeeze(newUf2)
            newWf = tf.squeeze(newWf)
            newUf1 = tf.reshape(newUf1,[1,5,5,1])
            newUf2 = tf.reshape(newUf2,[1,5,5,1])
            newWf = tf.reshape(newWf,[1,5,5,1])
            tf.summary.image('Uf1', newUf1)
            tf.summary.image('Uf2', newUf2)
            tf.summary.image('Wf', newWf)
            tf.summary.image('Bf', Bf)
        
        with tf.name_scope('4_output_gate'):
            newUo1 = tf.slice(Uo,[0,0,0,0],[5,5,1,1])
            newUo2 = tf.slice(Uo,[0,0,1,0],[5,5,1,1])
            newWo = tf.slice(Wo,[0,0,0,0],[5,5,1,1])
            newUo1 = tf.squeeze(newUo1)     #now a viewable [5x5] matrix
            newUo2 = tf.squeeze(newUo2)
            newWo = tf.squeeze(newWo)
            newUo1 = tf.reshape(newUo1,[1,5,5,1])
            newUo2 = tf.reshape(newUo2,[1,5,5,1])
            newWo = tf.reshape(newWo,[1,5,5,1])
            tf.summary.image('Uo1', newUo1)
            tf.summary.image('Uo2', newUo2)
            tf.summary.image('Wo', newWo)
            tf.summary.image('Bo', Bo)

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
    
    # Initial values of input-update wts.
    # save for plotting later
    Wts_array_initial = []
    Wts_array_initial.append(tf.squeeze(newU1).eval())
    Wts_array_initial.append(tf.squeeze(newU2).eval())
    Wts_array_initial.append(tf.squeeze(newW).eval())

    # Initial values of input-gate (g-gate) wts.
    gWts_array_initial = []
    gWts_array_initial.append(tf.squeeze(newUg1).eval())
    gWts_array_initial.append(tf.squeeze(newUg2).eval())
    gWts_array_initial.append(tf.squeeze(newWg).eval())

    # Initial values of forget-gate wts.
    fWts_array_initial = []
    fWts_array_initial.append(tf.squeeze(newUf1).eval())
    fWts_array_initial.append(tf.squeeze(newUf2).eval())
    fWts_array_initial.append(tf.squeeze(newWf).eval())
    
    # Initial values of output-gate (q-gate) wts.
    oWts_array_initial = []
    oWts_array_initial.append(tf.squeeze(newUo1).eval())
    oWts_array_initial.append(tf.squeeze(newUo2).eval())
    oWts_array_initial.append(tf.squeeze(newWo).eval())

# Below would only used to test if the input makes sense
#    output = sess.run(image)

    for step in range(NUM_TRAINING_STEPS): # 0 to 100
        _, l, predictions = sess.run([optimizer, loss, lstm_output])
        if step % 1 == 0:
            ms = sess.run(msumm) # merge summary
            writer.add_summary(ms, step)
        
        print("Step: ", step)
        print("Loss: ", loss.eval())
        print("New Loss: ", l)
        
    # Input-update weights
    print("Initial U1 wts: \n", Wts_array_initial[0])
    print("Initial U2 wts: \n", Wts_array_initial[1])
    print("Initial W  wts: \n", Wts_array_initial[2])
    Wts_array_final = []
    Wts_array_final.append(tf.squeeze(newU1).eval())
    Wts_array_final.append(tf.squeeze(newU2).eval())
    Wts_array_final.append(tf.squeeze(newW).eval())
    print("Final U1 wts: \n", Wts_array_final[0])
    print("Final U2 wts: \n", Wts_array_final[1])
    print("Final W wts:  \n", Wts_array_final[2])
    
    # Normalize colormap values for six arrays
    vmin = 1e40
    vmax = -1e40
    for i in range(3):
        dd = Wts_array_initial[i].ravel()
        vmin = min(vmin, np.min(dd))
        vmax = max(vmax, np.max(dd))
    for i in range(3):
        dd = Wts_array_final[i].ravel()
        vmin = min(vmin, np.min(dd))
        vmax = max(vmax, np.max(dd))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)        
    print("vmin = ", vmin)
    print("vmax = ", vmax)
    
    cmap = cm.Greys_r
#    cmap = cm.cool
    f, axarr = plt.subplots(2, 3)
    axarr[0, 0].imshow(Wts_array_initial[0], norm=norm, cmap=cmap)
    axarr[0, 1].imshow(Wts_array_initial[1], norm=norm, cmap=cmap)
    axarr[0, 2].imshow(Wts_array_initial[2], norm=norm, cmap=cmap)
    axarr[1, 0].imshow(Wts_array_final[0], norm=norm, cmap=cmap)
    axarr[1, 1].imshow(Wts_array_final[1], norm=norm, cmap=cmap)
    axarr[1, 2].imshow(Wts_array_final[2], norm=norm, cmap=cmap)
    axarr[0,0].set_title('init U1')
    axarr[0,1].set_title('init U2')
    axarr[0,2].set_title('init W')
    axarr[1,0].set_title('final U1')
    axarr[1,1].set_title('final U2')
    axarr[1,2].set_title('final W')
    plt.show()

    
    # G-gate weights (input gate)
    print("Initial Ug1 wts: \n", gWts_array_initial[0])
    print("Initial Ug2 wts: \n", gWts_array_initial[1])
    print("Initial Wg  wts: \n", gWts_array_initial[2])
    gWts_array_final = []
    gWts_array_final.append(tf.squeeze(newUg1).eval())
    gWts_array_final.append(tf.squeeze(newUg2).eval())
    gWts_array_final.append(tf.squeeze(newWg).eval())
    print("Final Ug1 wts: \n", gWts_array_final[0])
    print("Final Ug2 wts: \n", gWts_array_final[1])
    print("Final Wg wts:  \n", gWts_array_final[2])
    
    
    # Normalize colormap values for six arrays
    vmin = 1e40
    vmax = -1e40
    for i in range(3):
        dd = gWts_array_initial[i].ravel()
        vmin = min(vmin, np.min(dd))
        vmax = max(vmax, np.max(dd))
    for i in range(3):
        dd = gWts_array_final[i].ravel()
        vmin = min(vmin, np.min(dd))
        vmax = max(vmax, np.max(dd))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)        
    print("vmin = ", vmin)
    print("vmax = ", vmax)

    f, axarr = plt.subplots(2, 3)
    axarr[0, 0].imshow(gWts_array_initial[0], norm=norm, cmap=cmap)
    axarr[0, 1].imshow(gWts_array_initial[1], norm=norm, cmap=cmap)
    axarr[0, 2].imshow(gWts_array_initial[2], norm=norm, cmap=cmap)
    axarr[1, 0].imshow(gWts_array_final[0], norm=norm, cmap=cmap)
    axarr[1, 1].imshow(gWts_array_final[1], norm=norm, cmap=cmap)
    axarr[1, 2].imshow(gWts_array_final[2], norm=norm, cmap=cmap)
    axarr[0,0].set_title('init Ug1')
    axarr[0,1].set_title('init Ug2')
    axarr[0,2].set_title('init Wg')
    axarr[1,0].set_title('final Ug1')
    axarr[1,1].set_title('final Ug2')
    axarr[1,2].set_title('final Wg')
    plt.show()

    # F-gate weights
    print("Initial Uf1 wts: \n", fWts_array_initial[0])
    print("Initial Uf2 wts: \n", fWts_array_initial[1])
    print("Initial Wf  wts: \n", fWts_array_initial[2])
    fWts_array_final = []
    fWts_array_final.append(tf.squeeze(newUf1).eval())
    fWts_array_final.append(tf.squeeze(newUf2).eval())
    fWts_array_final.append(tf.squeeze(newWf).eval())
    print("Final Uf1 wts: \n", fWts_array_final[0])
    print("Final Uf2 wts: \n", fWts_array_final[1])
    print("Final Wf  wts: \n", fWts_array_final[2])
    
    # Normalize colormap values for six arrays
    vmin = 1e40
    vmax = -1e40
    for i in range(3):
        dd = fWts_array_initial[i].ravel()
        vmin = min(vmin, np.min(dd))
        vmax = max(vmax, np.max(dd))
    for i in range(3):
        dd = fWts_array_final[i].ravel()
        vmin = min(vmin, np.min(dd))
        vmax = max(vmax, np.max(dd))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)        
    print("vmin = ", vmin)
    print("vmax = ", vmax)

    f, axarr = plt.subplots(2, 3)
    axarr[0, 0].imshow(fWts_array_initial[0], norm=norm, cmap=cmap)
    axarr[0, 1].imshow(fWts_array_initial[1], norm=norm, cmap=cmap)
    axarr[0, 2].imshow(fWts_array_initial[2], norm=norm, cmap=cmap)
    axarr[1, 0].imshow(fWts_array_final[0],   norm=norm, cmap=cmap)
    axarr[1, 1].imshow(fWts_array_final[1],   norm=norm, cmap=cmap)
    axarr[1, 2].imshow(fWts_array_final[2],   norm=norm, cmap=cmap)
    axarr[0,0].set_title('init Uf1')
    axarr[0,1].set_title('init Uf2')
    axarr[0,2].set_title('init Wf')
    axarr[1,0].set_title('final Uf1')
    axarr[1,1].set_title('final Uf2')
    axarr[1,2].set_title('final Wf')
    plt.show()
    
    # Q-gate weights (output gate)
    print("Initial Uo1 wts: \n", oWts_array_initial[0])
    print("Initial Uo2 wts: \n", oWts_array_initial[1])
    print("Initial Wo  wts: \n", oWts_array_initial[2])
    oWts_array_final = []
    oWts_array_final.append(tf.squeeze(newUo1).eval())
    oWts_array_final.append(tf.squeeze(newUo2).eval())
    oWts_array_final.append(tf.squeeze(newWo).eval())
    print("Final Uo1 wts: \n", oWts_array_final[0])
    print("Final Uo2 wts: \n", oWts_array_final[1])
    print("Final Wo  wts: \n", oWts_array_final[2])
    
    # Normalize colormap values for six arrays
    vmin = 1e40
    vmax = -1e40
    for i in range(3):
        dd = oWts_array_initial[i].ravel()
        vmin = min(vmin, np.min(dd))
        vmax = max(vmax, np.max(dd))
    for i in range(3):
        dd = oWts_array_final[i].ravel()
        vmin = min(vmin, np.min(dd))
        vmax = max(vmax, np.max(dd))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)        
    print("vmin = ", vmin)
    print("vmax = ", vmax)
    
    f, axarr = plt.subplots(2, 3)
    axarr[0, 0].imshow(oWts_array_initial[0], norm=norm, cmap=cmap)
    axarr[0, 1].imshow(oWts_array_initial[1], norm=norm, cmap=cmap)
    axarr[0, 2].imshow(oWts_array_initial[2], norm=norm, cmap=cmap)
    axarr[1, 0].imshow(oWts_array_final[0], norm=norm, cmap=cmap)
    axarr[1, 1].imshow(oWts_array_final[1], norm=norm, cmap=cmap)
    axarr[1, 2].imshow(oWts_array_final[2], norm=norm, cmap=cmap)
    axarr[0,0].set_title('init Uo1')
    axarr[0,1].set_title('init Uo2')
    axarr[0,2].set_title('init Wo')
    axarr[1,0].set_title('final Uo1')
    axarr[1,1].set_title('final Uo2')
    axarr[1,2].set_title('final Wo')
    plt.show()
  






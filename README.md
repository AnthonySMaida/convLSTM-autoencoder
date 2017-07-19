# convLSTM-autoencoder
TensorFlow prototype for a convolutional LSTM autoencoder, inspired by Lotter, Krieman, Cox (ICLR 2017) "Deep predictive coding
networks for video prediction and unsupervised learning."

Release notes:
- Can read a jpeg image and resize it for input to the network.
- Have modified code to make sure tensor dimensions in the model match. Have checked using print statements.
- Have added TensorBoard calls to create and visualize the unrolled model graph. The model appears to be correct.
- Have added TensorBoard calls to image convolutions. This is giving useful debugging information.
- Added clip_by_global_norm() to manage exploading and vanishing gradients (helps a lot). Experimenting with exponential decay learning rate. Also has TensorBoard summaries for loss and learning rate.

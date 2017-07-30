# convLSTM-autoencoder
TensorFlow prototype for a convolutional LSTM autoencoder, inspired by Lotter, Krieman, Cox (ICLR 2017) "Deep predictive coding
networks for video prediction and unsupervised learning."

Release notes before July 30:
- Can read a jpeg image and resize it for input to the network.
- Have modified code to make sure tensor dimensions in the model match. Have checked using print statements.
- Have added TensorBoard calls to create and visualize the unrolled model graph. The model appears to be correct.
- Have added TensorBoard calls to image convolutions. This is giving useful debugging information.
- Added clip_by_global_norm() to manage exploading and vanishing gradients (helps a lot). Experimenting with exponential decay learning rate. Also has TensorBoard summaries for loss and learning rate.

July 30 release notes:
- ZK added tensorboard visualization of weight matrices.
- Added random number seeds to weight initializations so simulation runs are reproducible. Change seeds to change initialization.
- Found that the newly added tensorboard weight matrix visualizations were for some unknown reason erratic and unreliable. Additionally, noticed that visualization of the output gate weights suggested that there was no learning in this weight set at all. What really happened was that learning rescaled the weight values approximately equally and the color map re-normalization made it look like the weight values were not changing.
- Because of the above issue, added before-and-after training weight plots and printouts using matplotlib.
- Still need to add weight histogram and distribution plots for tensorboard.

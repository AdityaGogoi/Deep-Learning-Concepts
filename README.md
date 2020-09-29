# Deep-Learning-Concepts
This repository contains Jupyter notebooks with Deep Learning concepts gathered from the Udacity Deep Learning Nanodegree

The repository is divided into sections pertaining to a specific field of Deep learning. Each section will contain mostly Jupyter notebooks that demonstrate the implementation of a particular Deep Learning model for a particualr use-case. There might be supporting files like Python scripts and images to support the notebook as well.

Here are the sections with a description for each notebook contained in them:
## Section 2 - Deep Learning with PyTorch
* We will start building Neural Networks using the **PyTorch** library, which was created by the **Facebook AI Research Team** and is now open source.
* PyTorch is a great framework for deep learning and mainly works the **tensor** data structure. It just makes the entire process of creating multiple layers of Neural Networks a very simple process.

#### 1. Tensors in PyTorch
* We will introduce PyTorch, a framework for building and training Neural Networks.
* PyTorch uses tensors as its standard data structure, which is very similar to NumPy array.
* It makes manipulation and movement of large arrays really simple to execute. It also has modules for automatically calculating gradients and building neural networks.
* We will understand how tensors work and implement a standard neural network using PyTorch.

#### 2. Neural Networks in PyTorch
* Deep Learning Networks contain dozens to hundreds of Neural Network layers. Hence, implementing it manually is a very cumbersome task.
* That is why we have libraries like PyTorch, which we will use in this notebook to implement a Deep Learning Network.
* We will use this model to identify digits from the MNIST dataset.

#### 3. Training Neural Networks
* This notebook will explain the training process of a Neural Network.
* We will cover concepts like Types of Losses, Loss Function, Gradient Descent, and Backpropagation. We will also understand their role in the training of a Deep Neural Network.
* We will compare the prediction accuracy between a model that was naively trained from previous notebook and the current model whose training parameters are tuned.

#### 4. Fashion-MNIST
* The usual MNIST dataset is pretty trivial in terms of deep-learning networks to train on, and we can easily achieve accuracy grater than 97%.
* The Fashion-MNIST dataset set of 28x28 greyscale images of clothes. It is more complex than MNIST, so it is a better test of our Neural Networks performance and a better representation of real-world datasets.
* We will perform the same tasks as in the previous notebook, but our model architecture will change to accomodate the more complex dataset.

#### 5. Inference and Validation
* In this notebook, we will test our trained model via the validation step, where the model predicts on data it has never seen before.
* We will also cover concepts like overfitting, dropout and other terms that will help us to validate our model.

## Section 3 - Convolutional Neural Networks
* CNNs (or ConvNet) are a class of Deep Learning models that are mostly applied for analyzing visual data like images videos, etc.
* They usually consist of Multi-Layer Perceptrons (MLP), which are fully-connected Neral Networks, preceded by Convolution Layer(s), which recognizes the patterns in an image.

#### 1. MNIST Multi Layer Perceptron 
* In this example, we will train an MLP to classify hand-written digits from the MNIST dataset.
* The process is broken down in the following steps:
  1. We will load and visualize the data first to check for patterns and divide it into train and test dataset.
  2. We will then create a Neral Network.
  3. We will train our model on the training dataset.
  4. We will then evaluate our model on the test dataset.

#### 2. Finding Edges and Custom Kernels
* In this notebook, we will create our own convolutional filters and learn how they work on an image to obtain patterns.
* We will be using the OpenCV library to get a standard filter to detect horizontal and vertical edges. Then we will modify it and see if it improves the pattern gathering.

#### 3. Layer Visualizations
* In these notebooks, we will understand how the convolution and pooling layers work in a CNN. 
* We will visualize the outputs of these layers to understand what role they play in CNNs i.e. bringing out the patterns in an image and removing unnecessary data.

#### 4. CNNs in PyTorch
* We will be using CNN in PyTorch to classify images from the CIFAR-10 database.
* The images in this database include small color images that belong to one of ten classes.
* We will introduce CNNs in PyTorch by using a simple CNN model to train and predict on this database.
* Later we will visualize some of the output to see how accurate our predictions are.

#### 5. Transfer Learning
* We will use Transfer Learning to make our own model more accurate.
* We will load a pre-trained VGG Net, freeze its weights and add our own linear layer at the end of the model.
* Then we will train this model for a couple of epochs and then test its performance.

#### 6. Weight Initialization
* Here we will learn how crucial the initial weights are to a network for a profitable training cycle.
* We will try different ways of weight initialization for a CNN and see how it affects the training and final performance of the model. We will use the MNIST data for training.

#### 7. Linear Autoencoder
* In this notebook, we will define a linear autoencoder, which can be used for compression (i.e. representing a large raw data like an image in an efficient manner) and then reconstruction (i.e. recreating original raw data from compressed version).
* We will use the autoencoder to compress and then reconstruct images from the MNIST dataset.

#### 8. Convolutional Autoencoder
* We will improve upon the previous autoencoder by adding convolutional layers to it, so that the compression and reconstruction of images is of better quality. This is included in the **Convolutional Autoencoder** notebook.
* The transpose convolutional layers used in the **Convolutional Autoencoder** notebook can lead to undesired effects like checkerboard patterns. To avoid this, we can resize the layers using nearest neighbor or bilinear interpolation (i.e. upsampling) followed by convolutional layer. This is demonstrated in the **Upsampling** notebook.

#### 9. De-noising Autoencoder
* We will train a Convolutional Autoencoder to remove noise from images in the MNIST dataset.
* First, we will create copies of the images and add noise to it.
* Next, we will train the Autoencoder with the noisy images as input and clean images as targets.

#### 10. Style Transfer
* In this notebook, we will apply style transfer from one image into the content of other images.
* We will load in a pre-trained VGG Net and freeze select layers to be used as a feature-extractor.
* The we will load in the content and style images. We will use different layers of the model to extract content style from the respective images.
* After calculating the Gram Matrix for convolutional layers, we will start implementing the style on a target image, with Content, Style and Total Loss to check if we are getting an accurate Transfer.

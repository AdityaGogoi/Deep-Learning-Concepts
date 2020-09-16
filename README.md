# Deep-Learning-Concepts
This repository contains Jupyter notebooks with Deep Learning concepts gathered from the Udacity Deep Learning Nanodegree

The repository is divided into sections pertaining to a specific field of Deep learning. Each section will contain mostly Jupyter notebooks that demonstrate the implementation of a particular Deep Learning model for a particualr use-case. There might be supporting files like Python scripts and images to support the notebook as well.

Here are the sections with a description for each notebook contained in them:

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

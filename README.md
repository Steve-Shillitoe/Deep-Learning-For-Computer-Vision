# Deep-Learning-For-Computer-Vision
Deep Learning with Keras

This project contains Python modules written while following the Udemy online course **Python for Computer Vision with OpenCV and Deep Learning**. 

In the first module **1_Data_Classification.py**, Keras is used to create a machine learning module that can detect whether a banknote is a forgery or not.  The training data is numerical values in a CSV file.

In the second module **2_Keras_CNN_With_MNIST.py**, Keras is used to create a Convolutional Neural Network **CNN** to classify **grayscale** images of handwritten digits 0-9 in the MNIST dataset. 

In the third module **3_Keras_CNN_With_CIFAR_10.py** keras is used to classify **colour** images in the CIFAR-10 dataset. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.

In the fourth module **4_Keras_CNN_MNIST_Fashion_Dataset.py**  an image classifier is build using Keras and a convolutional neural network 
and applied to the Fashion MNIST dataset.  This dataset contains 70,000 28x28 grayscale images of fashion products from 10 categories 
from a dataset of Zalando article images, with 7,000 images per category.
The training set consists of 60,000 images and the test set consists of 10,000 images. 

In the fifth module **5_Deep_Learning_On_Custom_Images.py**  an image classifier is created using Keras and a convolutional neural network.  
It is used to classify images of cats & dogs in the form of  raw colour jpeg files of various sizes. 
In each convolutional layer, the size of each image is increased/decreased to the same size, 150x150 pixels

The ImageDataGenerator from the Keras preprocessing module is used to add random transformations to the 
cat & dog images to make the model training more challenging and thus make the model more robust. 

Due to the complexity of the images, the model would require 100 epochs to train.
So to save time, an existing model, cat_dog_100epochs.h5 is loaded & used to make
predictions. 

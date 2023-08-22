"""
The MNIST dataset contains grayscale 28x28 pixel images of handwritten single digits from 0-9.
As we are dealing with images, in this module a Convolutional Neural Network is built using Keras 
to classify the images of handwritten digits. 

The MNIST dataset has 60,000 training images and 10,000 test images.
Each array of images has 4 dimensions (60000, 28, 28, 1) or (image number, x, y, colour channel)

One-hot encoding is used for the labels. So instead of using labels such as 'one', 'two' etc, 
each image will have an array of 10 cells, where the label is represented by a one in its index position
and zeros in the other positions. So a drawn digit of 4 would have the following label array 
[0,0,0,0,1,0,0,0,0,0]
This works well with an output layer of 10 neurons where an activation function is chosen 
that fires either a 0 or a 1.
"""
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

(x_train, y_train),(x_test, y_test) = mnist.load_data()

single_image = x_train[0]
#plt.imshow(single_image, cmap='gray_r')
#plt.show()
#one hot encode
y_cat_test = to_categorical(y_test, 10)  #10 classes 0-9
y_cat_train = to_categorical(y_train, 10)  #10 classes 0-9
#print(y_cat_train[0]) 1 in the 6th position as expected

#Normalise image pixel values to 0-1
x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

print("Image data has no colour channel,  {} which is {} images, width of an image {}, height of an image {}"
      .format(x_train.shape, x_train.shape[0], x_train.shape[1], x_train.shape[2])
      )
#reshape image data to include a colour channel. 
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape)

#Create the model
model = Sequential()
#Convolutional layer with activation function Rectified Linear Unit
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28,28,1), activation='relu'))
#Pooling layer
model.add(MaxPool2D(pool_size=(2,2)))
#we need to take the above CNN layers and transform them into a form that can interact with a Dense layer
#2D to 1D
model.add(Flatten())
#Dense layer
model.add(Dense(units=128, activation='relu'))
#Output layer, the classifier
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())

#train the model
model.fit(x_train, y_cat_train, epochs=2)

model.evaluate(x_test, y_cat_test)

predictions = model.predict(x_test)
print("predictions={}".format(predictions))
# Convert probabilities to class labels using argmax
predicted_classes = np.argmax(predictions, axis=-1)

# Calculate the classification report
report = classification_report(y_test, predicted_classes)

print(report)











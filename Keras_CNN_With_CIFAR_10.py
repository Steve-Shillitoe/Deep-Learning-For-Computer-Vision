from keras.datasets import cifar10 #colour images, therefore 3 colour channels
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

(x_train, y_train),(x_test, y_test) = cifar10.load_data()
#Normalise image pixel values to 0-1
x_train = x_train / x_train.max()
x_test = x_test / x_test.max()
#one hot encode
y_cat_test = to_categorical(y_test, 10)  #10 classes 0-9
y_cat_train = to_categorical(y_train, 10)  #10 classes 0-9

model = Sequential()
#convolutional layer
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(32,32,3), activation='relu'  ))
#pooling layer reduces the spatial dimensions (width and height) of the input feature maps 
#while retaining the most important information
model.add(MaxPool2D(pool_size=(2,2)))
#a second convolutional layer
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(32,32,3), activation='relu'  ))
#a second pooling layer
model.add(MaxPool2D(pool_size=(2,2)))
#flatten  2D to 1D
model.add(Flatten())
#Dense output layer
model.add(Dense(units=256, activation='relu'))
#Classifier layer
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())

#train the model
model.fit(x_train, y_cat_train, verbose=1, epochs=10)

model.evaluate(x_test, y_cat_test)

predictions = model.predict(x_test)
print("predictions={}".format(predictions))
# Convert probabilities to class labels using argmax
predicted_classes = np.argmax(predictions, axis=-1)

# Calculate the classification report
report = classification_report(y_test, predicted_classes)

print(report)
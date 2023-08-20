from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Flatten, Conv2D, MaxPooling2D, Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

#load the data
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#single_image = x_train[0]
#plt.imshow(single_image) #, cmap=plt.cm.binary
#plt.show()

## Preprocess the data
#normalise the data
x_train = x_train/x_train.max()
x_test = x_test/x_test.max()
#print(x_train.shape, x_test.shape)

#reshape image data to include a colour channel. 
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#print(x_train.shape)

#Convert the y_train & y_test data to one hot encoded for categorical analysis by Keras
y_cat_test = to_categorical(y_test, 10)  #10 classes 0-9
y_cat_train = to_categorical(y_train, 10)  #10 classes 0-9

#Build the model
model = Sequential()
#2D convolutional layer
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28,28,1), activation='relu'))
#pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
#2D to 1D
model.add(Flatten())
#Dense layer
model.add(Dense(units=128, activation='relu'))
#Output layer, the classifier
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())

#train the model
model.fit(x_train, y_cat_train, epochs=10)

#evaluate the model
predictions = model.predict(x_test)
print("predictions={}".format(predictions))
# Convert probabilities to class labels using argmax
predicted_classes = np.argmax(predictions, axis=-1)

# Calculate the classification report
report = classification_report(y_test, predicted_classes)

print(report)









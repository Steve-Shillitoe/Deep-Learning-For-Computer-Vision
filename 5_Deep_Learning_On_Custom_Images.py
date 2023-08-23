"""
In this module an image classifier is created using Keras and a convolutional neural network.  
It is used to classify images of cats & dogs in the form of  raw colour jpeg files of various sizes. 
In each convolutional layer, the size of each image is increased/decreased to the 
same size, 150x150 pixels

The ImageDataGenerator from the Keras preprocessing module is used to add random transformations to the 
images to make the model training more challenging and thus make the model more robust. 

The cat & dog images are stored in the CATS_DOGS folder.

Due to the complexity of the images, the model would require 100 epochs to train.
So to save time, an existing model, cat_dog_100epochs.h5 is loaded & used to make
predictions. 

"""

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense
from keras.preprocessing import image


#Make the images none uniform with random transformations
image_gen = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1/255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')

#In each convolutional layer, the size of each image is increased/decreased to the 
#same size, 150x150 pixels
input_shape=(150,150,3)
model = Sequential()
#As the images are complex, add 3 convolutional layers with max pooling
#A filter also know as a kernel is a small matrix that is slid over the image matrix
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128))
model.add(Activation('relu'))

#randomly turn off 50% of the neurons to prevent overfitting
model.add(Dropout(0.5))

#Output layer, 1 neuron as the output is binary, 1 for cat, 0 for dog
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

batch_size = 16
train_image_gen = image_gen.flow_from_directory('CATS_DOGS/train',
                                                target_size=input_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory('CATS_DOGS/test',
                                                target_size=input_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

print(train_image_gen.class_indices)

#results = model.fit_generator(train_image_gen, epochs=2, steps_per_epoch=150,
#                              validation_data=test_image_gen, 
 #                             validation_steps=12)

#print(results.history['accuracy'])

#use existing model trained using 100 epochs
new_model = load_model('cat_dog_100epochs.h5')

#Test the model to see if it correctly identifies a dog
dog_file = 'CATS_DOGS/test/DOG/10008.jpg'
dog_img = image.load_img(dog_file, target_size=(150,150))
dog_img = image.img_to_array(dog_img)
dog_img = np.expand_dims(dog_img, axis=0) #(150,150,3) -> (1, 150,150,3)
#normal pixel values
dog_img = dog_img/255

result =new_model.predict(dog_img)
print("result={}".format(result))

# Convert probabilities to class labels using argmax
predicted_classes = np.argmax(result, axis=-1)

# Calculate the classification report
#report = classification_report(y_test, predicted_classes)

print(predicted_classes)

















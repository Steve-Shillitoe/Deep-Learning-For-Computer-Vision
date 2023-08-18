#from msilib import Feature
import numpy as np
from numpy import genfromtxt

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# For data preprocessing and splitting
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


data = genfromtxt('bank_note_data.txt', delimiter=',')
labels = data[:,4] #last column
features = data[:,0:4]
X = features
y = labels
#print(data)
#print(labels)
#print(features)

# Split dataset into train and test sets (using a 67-33 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Feature scaling helps to normalize the range of independent variables (features) in a dataset. 
#It transforms the features to a similar scale, which can improve the performance and 
#convergence of various machine learning algorithms.
from sklearn.preprocessing import MinMaxScaler
scaler_object = MinMaxScaler()
scaler_object.fit(X_train)  #fit to training data only, don't want the scaler object to see test data
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

#set up model
model = Sequential()
#The Dense layer in Keras is used to create fully connected (dense) neural network layers.
#The input_dim parameter specifies the number of input units/neurons in the layer, 
#which is required when defining the first layer of your neural network.
#The parameter "units" refers to the number of artificial neurons (also known as nodes or processing elements)
# in a particular layer.
#An activation function is applied to the output of a neuron, 
#to determines if the neuron should "fire" (i.e., produce an output signal) based on the input it receives. 
#ReLU (Rectified Linear Activation): This is one of the most widely used activation functions. 
#It sets negative inputs to zero and passes positive inputs as they are.
#It helps alleviate the vanishing gradient problem and speeds up training.
#Define input layer
model.add(Dense(units=4, input_dim=4, activation='relu')) #4 units because data array has 4 columns
#Define a hidden layer
model.add(Dense(units=8, activation='relu'))
#Output layer
#Only one nueron, as the output is binary. 
#Sigmoid Activation (logistic function): This function squashes the input into the range (0, 1),
# making it suitable for binary classification problems where the output needs to represent probabilities.
model.add(Dense(units=1, activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the model
model.fit(scaled_X_train, y_train, epochs=50, verbose=2)

predictions = model.predict(scaled_X_test)
print("predictions={}".format(predictions))
# Convert probabilities to class labels using argmax
predicted_classes = np.argmax(predictions, axis=-1)

# Calculate the classification report
report = classification_report(y_test, predicted_classes)

print(report)

#model.save('myModel.h5')
#newModel = load_model('myModel.h5')












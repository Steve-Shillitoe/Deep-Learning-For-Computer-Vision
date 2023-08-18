from msilib import Feature
import numpy as np
from numpy import genfromtxt

data = genfromtxt('bank_note_data.txt', delimiter=',')
labels = data[:,4] #last column
features = data[:,0:4]
X = features
y = labels
#print(data)
#print(labels)
#print(features)

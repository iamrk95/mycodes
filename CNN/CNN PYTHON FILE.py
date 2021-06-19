###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>CNN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<###
#load the packages
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets,layers,models

#load the dataset
(xtrain,ytrain),(xtest,ytest) = datasets.cifar10.load_data()

#inspect the size
xtrain.shape

ytrain.shape

xtest.shape

ytrain[:5]

#reshape the data 
ytrain = ytrain.reshape(-1,)
ytrain

#create the list of the classes
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

plot_sample(xtrain, ytrain, 0)

plot_sample(xtrain, ytrain, 5)

#normalization
xtrain = xtrain / 255.0
xtest = xtest / 255.0

#build the CNN model
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#compile the model
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(xtrain, ytrain, epochs=10)

cnn.evaluate(xtest,ytest)

#data augmentation
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

plt.axis('off')
plt.imshow(xtrain[0])

plt.axis('off')
plt.imshow(data_augmentation(xtrain)[0])

#CNN model with data aumentation
cnn = models.Sequential([
    data_augmentation,
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train accuracy
cnn.fit(xtrain, ytrain, epochs=30)

#test accuracy
cnn.evaluate(xtest,ytest)
#######################################################################################

import os
import cv2
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Define your data directories and other constants
batch_size = 10
epochs = 10
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train/=255
x_test/=255

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Saving model...")
model.save("Model1_upgraded.h5")
print("Done.")

# Save the best model
# model.save("BestModel.h5")
# inputs = Input(shape=(9, 9, 1))
# flatten = Flatten()(inputs)
# dense = Dense(16, activation="relu")(flatten)
# dense = Dense(16, activation="relu")(dense)
# prediction = Dense(10, activation="softmax")(dense)
# model = EvolModel(inputs=inputs, outputs=prediction)
# myopt = NGA(population_size=29, sigma_init=15)
# model.compile(optimizer=myopt, loss="categorical_crossentropy", metrics=["accuracy"])
#
# model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test))
#
# # Evaluate the best model
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# print("Saving model...")
# model.save("Model1.h5")
# print("Done.")
#
# # Save the best model
# model.save("BestModel.h5")

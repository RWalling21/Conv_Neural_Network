import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pickle
import time

# --- TODO ---
# This section is more about what we need to learn to understand the NN better, and boy do we have a lot

# How does a ConvNN Learn?
# What exactly is Kernal Size and why is it important? 
# How does Pooling work
# What is Input Shape and how is conv shape defined
# What is Conv2D padding and how can it improve my model
# What is a Kernel Initializer and what is it used for
# Why do we use Sigmoid for our Output layer instead of Rectified Linear?
# What is Binary Crossentropy?
# Learn more about different conv architectures other than VGG
# What does the metrics property in the Compile function do

# --- In Progress ---

# use ImageDataGenerator to implement data augmentation

# --- Variables ---

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = np.array(X/255.0)
y = np.array(y)

# --- Model ---

model_Name = "VGG-convNN-{}".format(int(time.time())) #Formats the name of the Network to show which is most effective
tensorboard = TensorBoard(log_dir="logs\\{}".format(model_Name))
print(model_Name)

model = Sequential()

# Input Layer

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(75, 75, 1))) # Set's up first Conv layer with a kernel size of 3 - 3
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Hidden Layers

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

# Output Layer

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, batch_size=32, epochs=35, validation_split=0.2, callbacks=[tensorboard])
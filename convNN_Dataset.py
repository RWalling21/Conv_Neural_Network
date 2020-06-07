import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import cv2
import os

PATH = r"C:\Users\Admin\Desktop\PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 75
training_Data = []

def Format_Dataset():
    for category in CATEGORIES:
        img_Dir = os.path.join(PATH, category) # Joins the folder to Cat or Dog
        class_number = CATEGORIES.index(category)
        for i in os.listdir(img_Dir):
            try:
                # We use cv2 to convert the image color and make the image usable for other libraries
                img = cv2.imread(os.path.join(img_Dir, i), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                training_Data.append([img_array, class_number])
            except Exception as e:
                pass

def Pack_Dataset():
    X = []
    y = []

    random.shuffle(training_Data)
    for feature, label in training_Data:
        X.append(feature)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # Num of features (-1 wildcard), Image size, color

    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    
    print("Dataset saved")

Format_Dataset()
print("Formatted dataset with " + str(len(training_Data)) + " features")
Pack_Dataset()
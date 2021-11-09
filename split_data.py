import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
import shutil
import os, random, sys
import datetime
import csv
import glob

from sklearn.model_selection import train_test_split

# dataset names
dataset = glob.glob(f'{TRAINING_DIR}/*/*')
print(dataset[11029].split('/'))
# extract labels
X, y = [], []
for data in dataset:
    label = 1 if data.split('/')[2] == 'df' else 0
    y.append(label)
    #img = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(data)[:,:,::-1]
    img = cv2.resize(img, (150,150), interpolation=cv2.INTER_AREA)
    X.append(img)

X = np.array(X).reshape(-1,150,150,3)
y = to_categorical(y,2)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
x_train_file, y_train_file = open("arrays/x_train.npy","ba+"), open("arrays/y_train.npy","ba+")
x_test_file, y_test_file = open("arrays/x_test.npy","ba+"), open("arrays/y_test.npy","ba+")
np.save(x_train_file, x_train); np.save(y_train_file, y_train)
np.save(x_test_file, x_test); np.save(y_test_file, y_test)
x_train_file.close(); y_train_file.close()
x_test_file.close(); y_test_file.close()
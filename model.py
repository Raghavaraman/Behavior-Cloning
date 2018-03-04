
# coding: utf-8


import csv
import cv2
import pandas as pd
import numpy as np
import tqdm as tqdm
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from preprocess import preprocess_image, brightness_adjust

#Return the images and steering of Udacity data (Not to be confused with generator and yeild)

def Generator(lines, start, end):
    images = []
    steer_angles = []
    for index, row in lines.loc[start:end].iterrows():    
        source_path = lines['center'][index]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        steer_angle = float(lines['steering'][index])
        images.append(image)
        steer_angles.append(steer_angle)   
        source_path = lines['left'][index]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        steer_angle = float(lines['steering'][index])+0.25
        images.append(image)
        steer_angles.append(steer_angle) 
        flipable = np.random.random()
        source_path = lines['right'][index]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        steer_angle = float(lines['steering'][index])-0.25
        images.append(image)
        steer_angles.append(steer_angle)
    
    return images,steer_angles
        

#Help to get my driving data (Not a yeild generator)
    
    
def Generator1(lines, start, end):
    images = []
    steer_angles = []
    for index, row in lines.loc[start:end].iterrows():    
        col = np.random.choice(['center','left','right'])
        source_path = lines[col][index]
        filename = source_path.split('\\')[-1]
        current_path = './alex/IMG/' + filename   
        if col == 'left':
            steer_angle = float(lines['steering'][index])+0.25
        elif col == 'center':
            steer_angle = float(lines['steering'][index])
        else:
            steer_angle = float(lines['steering'][index])-0.25
        image1 = cv2.imread(current_path)
        image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
        image1 = preprocess_image(image1)
        images.append(image1)
        steer_angles.append(steer_angle)
    
    return images,steer_angles
        


#array for images and steering angle

images = []
steer = []

#Import Udacity data

data_frame = pd.read_csv('data/driving_log.csv', usecols=[0, 1, 2, 3])
data_frame = data_frame.sample(frac=1).reset_index(drop=True)
images , steer = Generator(data_frame,0, data_frame.shape[0])

#Import my Data

data_frame = pd.read_csv('alex/driving_log.csv', usecols=[0, 1, 2, 3])
data_frame = data_frame.sample(frac=1).reset_index(drop=True)
images1 , steer1 = Generator1(data_frame,0, data_frame.shape[0])

for i in range (0,len(images1)):
    images.append(images1[i])
    steer.append(steer1[i])


X_images = np.array(images)
Y_steer = steer

#The Neural network


model = Sequential()
model.add(Convolution2D(32, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="valid"))
model.add(ELU())
model.add(Dropout(.5))
model.add(Convolution2D(16, 3, 3, border_mode="valid"))
model.add(MaxPooling2D((2, 2), border_mode='valid'))
model.add(ELU())
model.add(Flatten())
model.add(Dense(64))
model.add(ELU())
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
model.summary()


model.fit(x=X_images, y=steer, batch_size=32, nb_epoch=15, validation_split=0.2, shuffle=True, verbose = 2)


print("Saving model weights and configuration file.")
model.save('model.h5')  # always save your weights after training or during training


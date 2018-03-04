
# coding: utf-8

# Since its coman for both model.py and drive.py i have written it separately and imported 

import cv2
import numpy as np

def preprocess_image(image):
# Cropping and resizing
    image = image[55:135, :, :]
    image = cv2.resize(image, (64,64))
    image = image.astype(np.float32)
 # Normalize image
    image = image/255.0 - 0.5
    return image


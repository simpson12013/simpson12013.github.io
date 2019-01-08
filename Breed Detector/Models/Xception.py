# Dependencies
import matplotlib.pyplot as plt

import os
import numpy as np
import tensorflow as tf

import keras
from keras.preprocessing import image
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)


# Load the Xception model
model = Xception(
    include_top=True,
    weights='imagenet')

# Default Image Size for Xception
image_size = (299, 299)


# Reusable function to call on given photo
def predict(image_path):
    """Use Xception to label image"""
    img = image.load_img(image_path, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    plt.imshow(img)
    print('Predicted:', decode_predictions(predictions, top=3)[0])


image_path = os.path.join("..", "Images", "Dog2.jpg")
predict(image_path)
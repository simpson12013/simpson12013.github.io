# Dependencies
import matplotlib.pyplot as plt

import os
import numpy as np
import tensorflow as tf

import keras
from keras.preprocessing import image
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
from keras.applications.vgg19 import (
    VGG19, preprocess_input, decode_predictions)

import cv2                 
from random import shuffle 
from tqdm import tqdm      

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from keras import backend as K
import tensorflow as tf
Vmodel = None
Xmodel = None
model = None
graph = None



# model load function ---------------------------------------------------------------------------------------------------------------------

def load_Vmodel():
    global Vmodel
    Vmodel = VGG19(
        include_top=True,
        weights='imagenet')

    # this is key : save the graph after loading the model
    global graph
    graph = tf.get_default_graph()



def load_Xmodel():
    global Xmodel
    Xmodel = Xception(
        include_top=True,
        weights='imagenet')

    # this is key : save the graph after loading the model
    global graph
    graph = tf.get_default_graph()


# --------------------------------------------------------------------------------------------------------------------------------------------





# VGG19 -------------------------------------------------------------------------------------------------------------------------------------

# Load the VGG19 model


# Default Image Size for VGG19
VGG_image_size = (224, 224)

# Refactor above steps into reusable function
def Vpredict(image_path):
    """Use VGG19 to label image"""
    K.clear_session()
    load_Vmodel()
    img = image.load_img(image_path, target_size=VGG_image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    Vmodel._make_predict_function()
    with graph.as_default():
        predictions = Vmodel.predict(x)
    print(predictions)
    plt.imshow(img)
    return jsonify('Predicted:', decode_predictions(predictions, top=3)[0])
    K.clear_session()






# /VGG19 ------------------------------------------------------------------------------------------------------------------------------------








# Xception ----------------------------------------------------------------------------------------------------------------------------------

# Load the Xception model

# Default Image Size for Xception
Xception_image_size = (299, 299)


# Reusable function to call on given photo
def Xpredict(image_path):
    """Use Xception to label image"""
    K.clear_session()
    load_Xmodel()
    img = image.load_img(image_path, target_size=Xception_image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    Xmodel._make_predict_function()
    global graph
    with graph.as_default():
        predictions = Xmodel.predict(x)
    print(predictions)
    plt.imshow(img)
    return jsonify('Predicted:', decode_predictions(predictions, top=3)[0])



# /Xception -------------------------------------------------------------------------------------------------------------------------------------



# Tensorflow -----------------------------------------------------------------------------------------------------------------------------------

TRAIN_DIR = '../Image_Data/train'
TEST_DIR = '../Image_Data/test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'model'
# just so we remember which saved model is which, sizes must match

def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = np.load('train_data.npy')


tf.reset_default_graph()
def train():
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    global model
    model = tflearn.DNN(convnet, tensorboard_dir='log')



    if os.path.exists('C:/Users/H/Desktop/KaggleDogsvsCats/{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    train = train_data[:-500]
    test = train_data[-500:]

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    
    

def run():

    test_data = np.load('test_data.npy')


    for num,data in enumerate(test_data[:1]):
        # cat: [1,0]
        # dog: [0,1]
        
        img_num = data[1]
        img_data = data[0]
        
        
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
        print(np.argmax(model_out))
        if np.argmax(model_out) == 1: Vpredict('../Image_Data/test/1.jpg')
        else: Xpredict('../Image_Data/test/1.jpg')
            
train()
run()


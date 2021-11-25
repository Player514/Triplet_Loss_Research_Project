import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import dicom_reader as dr
import cv2
from tensorflow.keras.preprocessing import image
from more_itertools import peekable
import tensorflow as tf

BATCH_SIZE = 10
STEPS = 5 
SHAPE = (224,224)
#annoying warning removal
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#images = dr.readall(reshape=(224,224))


#inp = preprocess_input(np.expand_dims(next(images), axis=0))

#assert(inp.shape == (1,224,224,3))

#resnet = ResNet50(weights='imagenet')

#print(img.shape)
#result = resnet.predict(inp)

#data = dr.tf_data(SHAPE)

resnet = ResNet50(weights='imagenet')

#result = resnet.predict(data.batch(BATCH_SIZE),steps=STEPS)

def run_resnet(data):
    return resnet.predict(data.batch(BATCH_SIZE),steps=STEPS)

    


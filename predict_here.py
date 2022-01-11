# The script that loads the model makes the required predictions

from smooth_tiled_predictions import predict_img_with_smooth_windowing
import smooth_tiled_predictions
import os
import shutil
import glob
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import callbacks, optimizers
from tensorflow.python.keras.models import Model, load_model, model_from_json
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import backend as K
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
# import segmentation_models as sm
import numpy as np
import pandas as pd
import cv2

import sys
sys.path.append('Extra_py_files')


K.set_image_data_format('channels_last')


# Loss and Accuracy functions used for the model training
def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# Main functions that loads the model divides the image into window_size, does all the r=predicting and
# merges all the predicted images into one final image.
def predict_with_image(img, model):
    # model = keras.models.load_model('Models\mobilenet.h5',
    #                                 custom_objects={
    #                                     # "weighted_loss":weighted_lyoss,
    #                                     "dice_coef": dice_coef,
    #                                     "dice_coef_loss": dice_coef_loss
    #                                 })
    input_image = preprocess_input(img)

    window_size = 256
    nb_classes = 1

    predictions_smooth = predict_img_with_smooth_windowing(
        input_image,
        window_size=window_size,
        # Minimal amount of overlap for windowing. Must be an even number.
        subdivisions=2,
        nb_classes=nb_classes,
        pred_func=(
            lambda img_batch_subdiv: model.predict(img_batch_subdiv)
        ))

    return tf.keras.preprocessing.image.array_to_img(predictions_smooth)

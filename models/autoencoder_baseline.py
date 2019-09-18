# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:36:30 2019

Autoencoder Baseline model for the musical images project. (v2)
This version considers each color layer indenpendently through a separate network

Last Modified: 17/8/2019

@author: Rivan
"""

import keras
from keras.datasets import cifar10
from keras.models import Model, load_model
from keras.layers import Dense, BatchNormalization, Input, Dropout
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from .. import models
import numpy as np
import matplotlib.pyplot as plt
from .. import model_param as c
import os
import time

print(keras.__version__)
start_time = time.time()

# Checkpoint Directory
# saveDir = "C:/Users/Rivan/Desktop/ptamarix/checkpoints/"
saveDir = os.getcwd() + '\\ptamarix\\checkpoints\\'
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)
    

def define_model():
    """ network architecture:
        vanilla autoencoder with fully-connected layer """
        
    input_img = Input(shape=(c.input_layer,))
    
    # Encoder
    x = Dense(units = c.hid_layer1, activation='relu')(input_img)
    x = Dense(units = c.hid_layer2, activation='relu')(x)
    x = Dense(units = c.hid_layer3, activation='relu')(x)
    
    # Bottleneck
    encoded = Dense(units = c.hid_layer4, activation='relu',
                    kernel_regularizer = l2(c.l2param), activity_regularizer = l1(c.l1param))(x)
    
    # Decoder
    x = Dense(units = c.hid_layer5, activation = 'relu')(encoded)
    x = Dense(units = c.hid_layer6, activation='relu')(x)
    x = Dense(units = c.hid_layer7, activation='relu')(x)
    decoded = Dense(units = c.output_layer, activation='sigmoid')(x)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    return autoencoder, encoder


    
if __name__ == '__main__':
    # Load dataset
    (xtrain, _), (xtest, _) = cifar10.load_data()

    # Preprocessing
    xtrain = xtrain.astype('float32') / 255.
    xtest = xtest.astype('float32') / 255.
    
    xtrain_red = xtrain[:,:,:,0]
    xtrain_green = xtrain[:,:,:,1] 
    xtrain_blue = xtrain[:,:,:,2]
    
    xtest_red = xtest[:,:,:,0]
    xtest_green = xtest[:,:,:,1] 
    xtest_blue = xtest[:,:,:,2]
    
    xtrain_red = xtrain_red.reshape(len(xtrain_red), np.prod(xtrain_red.shape[1:]))
    xtrain_green = xtrain_green.reshape(len(xtrain_green), np.prod(xtrain_green.shape[1:]))
    xtrain_blue = xtrain_blue.reshape(len(xtrain_blue), np.prod(xtrain_blue.shape[1:]))
    
    xtest_red = xtest_red.reshape(len(xtest_red), np.prod(xtest_red.shape[1:]))
    xtest_green = xtest_green.reshape(len(xtest_green), np.prod(xtest_green.shape[1:]))
    xtest_blue = xtest_blue.reshape(len(xtest_blue), np.prod(xtest_blue.shape[1:]))
    
    print("Train Dataset: ", xtrain_red.shape,"\nTest Dataset: ", xtest_red.shape)
    
    # Define Models
    autoencoder_red, encoder_red = define_model()
    autoencoder_green, encoder_green = define_model()
    autoencoder_blue, encoder_blue = define_model()
    
    autoencoder_red.compile(optimizer= Adam(lr = c.learning_rate), loss= c.loss_fx)
    autoencoder_green.compile(optimizer= Adam(lr = c.learning_rate), loss= c.loss_fx)
    autoencoder_blue.compile(optimizer= Adam(lr = c.learning_rate), loss= c.loss_fx)
    
    # Display Model
    plot_model(autoencoder_red, to_file='autoenocoder_baseline_red.png')
    autoencoder_red.summary()
    

    
    # train the model
    # RED
    es_cb_r = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
    chkpt_r = saveDir + 'AutoEncoder_baseline_2_red_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    cp_cb_r = ModelCheckpoint(filepath = chkpt_r, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    history_red = autoencoder_red.fit(xtrain_red, xtrain_red,
                              epochs= c.epochs,
                              batch_size= c.batch_size,
                              shuffle=True,
                              verbose = 1,
                              validation_data=(xtest_red, xtest_red),
                              callbacks = [es_cb_r, cp_cb_r])
    
    models.utils.display_history(history_red)
    
    
    # GREEN
    es_cb_g = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
    chkpt_g = saveDir + 'AutoEncoder_baseline_2_green_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    cp_cb_g = ModelCheckpoint(filepath = chkpt_g, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    history_green = autoencoder_green.fit(xtrain_green, xtrain_green,
                              epochs= c.epochs,
                              batch_size= c.batch_size,
                              shuffle=True,
                              verbose = 1,
                              validation_data=(xtest_green, xtest_green),
                              callbacks = [es_cb_g, cp_cb_g])
    
    models.utils.display_history(history_green)
    
    
    # BLUE
    es_cb_b = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
    chkpt_b = saveDir + 'AutoEncoder_baseline_2_blue_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    cp_cb_b = ModelCheckpoint(filepath = chkpt_b, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    history_blue = autoencoder_blue.fit(xtrain_blue, xtrain_blue,
                              epochs= c.epochs,
                              batch_size= c.batch_size,
                              shuffle=True,
                              verbose = 1,
                              validation_data=(xtest_blue, xtest_blue),
                              callbacks = [es_cb_b, cp_cb_b])
    
    models.utils.display_history(history_blue)
    
    
    counter = 0
    # show the result at the decoder output
    decoded_imgs_red = autoencoder_red.predict(xtest_red)
    decoded_imgs_green = autoencoder_green.predict(xtest_green)
    decoded_imgs_blue = autoencoder_blue.predict(xtest_blue)
    
    models.utils.display_model_evaluation(xtest_red, xtest_green, xtest_blue,
                                          decoded_imgs_red, decoded_imgs_green, decoded_imgs_blue, counter = counter)
    
    encoded_imgs_red = encoder_red.predict(xtest_red)
    encoded_imgs_green = encoder_green.predict(xtest_green)
    encoded_imgs_blue = encoder_blue.predict(xtest_blue)
    
    models.utils.display_bottleneck(encoded_imgs_red, counter = counter)
    models.utils.display_bottleneck(encoded_imgs_red, counter = counter)
    models.utils.display_bottleneck(encoded_imgs_red, counter = counter)
    
    end_time = time.time()
    
    print("Time Elapsed: ", end_time - start_time, "s")
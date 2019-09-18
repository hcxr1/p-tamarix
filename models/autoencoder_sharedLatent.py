# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:38:48 2019

Autoencoder shared latent model for the musical images project.

Last Modified: 18/8/2019

@author: Rivan
"""

import keras
from keras.datasets import cifar10
from keras.models import Model, load_model
from keras.layers import Dense, BatchNormalization, Input, Dropout, Concatenate
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from .. import model_param as c
from .. import models
import os
import time

print(keras.__version__)
start_time = time.time()

# Checkpoint Directory
saveDir = "C:/Users/Rivan/Desktop/ptamarix/checkpoints/"
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)
    

def define_model():
    """ network architecture:
        vanilla autoencoder with fully-connected layer """
        
    input_img_red = Input(shape=(c.input_layer,))
    input_img_green = Input(shape=(c.input_layer,))
    input_img_blue = Input(shape=(c.input_layer,))
    
    # Encoder
    x = Dense(units = c.hid_layer1, activation='relu')(input_img_red)
    x = Dense(units = c.hid_layer2, activation='relu')(x)
    x = Dense(units = c.hid_layer3, activation='relu')(x)
    
    y = Dense(units = c.hid_layer1, activation='relu')(input_img_green)
    y = Dense(units = c.hid_layer2, activation='relu')(y)
    y = Dense(units = c.hid_layer3, activation='relu')(y)
    
    z = Dense(units = c.hid_layer1, activation='relu')(input_img_blue)
    z = Dense(units = c.hid_layer2, activation='relu')(z)
    z = Dense(units = c.hid_layer3, activation='relu')(z)
    
    # Bottleneck for learning shared representation
    bottleneck = Concatenate([x, y, z])
    encoded = Dense(units = c.hid_layer4, activation='relu',
                    kernel_regularizer = l2(c.l2param), activity_regularizer = l1(c.l1param))(bottleneck)
    
    # Decoder
    x = Dense(units = c.hid_layer5, activation = 'relu')(encoded)
    x = Dense(units = c.hid_layer6, activation='relu')(x)
    x = Dense(units = c.hid_layer7, activation='relu')(x)
    
    y = Dense(units = c.hid_layer5, activation = 'relu')(encoded)
    y = Dense(units = c.hid_layer6, activation='relu')(y)
    y = Dense(units = c.hid_layer7, activation='relu')(y)
    
    z = Dense(units = c.hid_layer5, activation = 'relu')(encoded)
    z = Dense(units = c.hid_layer6, activation='relu')(z)
    z = Dense(units = c.hid_layer7, activation='relu')(z)
    
    decoded_red = Dense(units = c.output_layer, activation='sigmoid')(x)
    decoded_green = Dense(units = c.output_layer, activation='sigmoid')(y)
    decoded_blue = Dense(units = c.output_layer, activation='sigmoid')(z)
    
    autoencoder = Model([input_img_red, input_img_green, input_img_blue], [decoded_red, decoded_green, decoded_blue])
    encoder = Model([input_img_red, input_img_green, input_img_blue], encoded)

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
    
    # Define Model
    autoencoder, encoder = define_model()
    
    autoencoder.compile(optimizer= Adam(lr = c.learning_rate), loss= c.loss_fx)
    plot_model(autoencoder, to_file='autoenocoder_sharedLatent.png')
    autoencoder.summary()
    
    es_cb = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
    chkpt = saveDir + 'AutoEncoder_sharedLatent_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    # train the model
    history = autoencoder.fit([xtrain_red,xtrain_green,xtrain_blue], [xtrain_red,xtrain_green,xtrain_blue],
                              epochs= c.epochs,
                              batch_size= c.batch_size,
                              shuffle=True,
                              verbose = 1,
                              validation_data=([xtest_red, xtest_green, xtest_blue], [xtest_red, xtest_green, xtest_blue]),
                              callbacks = [es_cb, cp_cb])
    
    models.utils.display_history(history)
    
    counter = 0
    # show the result at the decoder output
    decoded_imgs_red = autoencoder.predict(xtest_red)
    decoded_imgs_green = autoencoder.predict(xtest_green)
    decoded_imgs_blue = autoencoder.predict(xtest_blue)
    
    models.utils.display_model_evaluation(xtest_red, xtest_green, xtest_blue,
                             decoded_imgs_red, decoded_imgs_green, decoded_imgs_blue, counter = counter)
    
    encoded_imgs_red = encoder.predict(xtest_red)
    encoded_imgs_green = encoder.predict(xtest_green)
    encoded_imgs_blue = encoder.predict(xtest_blue)
    
    models.utils.display_bottleneck(encoded_imgs_red, counter = counter)
    models.utils.display_bottleneck(encoded_imgs_red, counter = counter)
    models.utils.display_bottleneck(encoded_imgs_red, counter = counter)
    
    end_time = time.time()
    
    print("Time Elapsed: ", end_time - start_time, "s")
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:19:09 2019

Autoencoder Baseline model for the musical images project.

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
import numpy as np
import matplotlib.pyplot as plt
from .. import model_param as p
from .. import models
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
        
    input_img = Input(shape=(p.input_layer,))
    
    # Encoder
    x = Dense(units = p.hid_layer1, activation='relu')(input_img)
    x = Dense(units = p.hid_layer2, activation='relu')(x)
    x = Dense(units = p.hid_layer3, activation='relu')(x)
    
    # Bottleneck
    encoded = Dense(units = p.hid_layer4, activation='relu',
                    kernel_regularizer = l2(p.l2param), activity_regularizer = l1(p.l1param))(x)
    
    # Decoder
    x = Dense(units = p.hid_layer5, activation = 'relu')(encoded)
    x = Dense(units = p.hid_layer6, activation='relu')(x)
    x = Dense(units = p.hid_layer7, activation='relu')(x)
    decoded = Dense(units = p.output_layer, activation='sigmoid')(x)

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

    # Create new list of datasets (dimension train = 150000 * 1024, test = 30000 * 1024)
    train_dset = []
    train_dset.extend(xtrain_red)
    train_dset.extend(xtrain_green)
    train_dset.extend(xtrain_blue)
    dset_train = np.asarray(train_dset)
    
    test_dset = []
    test_dset.extend(xtest_red)
    test_dset.extend(xtest_green)
    test_dset.extend(xtest_blue)
    dset_test = np.asarray(test_dset)  
    
    print("Train Dataset: ", dset_train.shape,"\nTest Dataset: ", dset_test.shape)
    
    # Define Model
    autoencoder, encoder = define_model()
    
    autoencoder.compile(optimizer= Adam(lr = p.learning_rate), loss= p.loss_fx)
    plot_model(autoencoder, to_file='autoenocoder.png')
    autoencoder.summary()
    
    es_cb = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
    chkpt = saveDir + 'AutoEncoder_baseline_1_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    # train the model
    history = autoencoder.fit(dset_train, dset_train,
                              epochs= p.epochs,
                              batch_size= p.batch_size,
                              shuffle=True,
                              verbose = 1,
                              validation_data=(dset_test, dset_test),
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
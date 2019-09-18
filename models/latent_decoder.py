# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:01:11 2019

Decoder model to extract the output color (sharedLatent and simple NLP)

Last Modified: 20/8/2019

@author: Rivan
"""
import sys
sys.path.append('..')


from keras.layers import Input, Dense 
from keras.models import load_model, Model
import numpy as np
import model_param as mp



def define_decoder():
    """
        Define decoder model that takes bottleneck layer as the input
    """

    encoding = Input(shape = (mp.hid_layer4,))
    # RED
    x = Dense(units = mp.hid_layer5, activation = 'relu')(encoding)
    x = Dense(units = mp.hid_layer6, activation='relu')(x)
    x = Dense(units = mp.hid_layer7, activation='relu')(x)
    
    # GREEN
    y = Dense(units = mp.hid_layer5, activation = 'relu')(encoding)
    y = Dense(units = mp.hid_layer6, activation='relu')(y)
    y = Dense(units = mp.hid_layer7, activation='relu')(y)
    
    # BLUE
    z = Dense(units = mp.hid_layer5, activation = 'relu')(encoding)
    z = Dense(units = mp.hid_layer6, activation='relu')(z)
    z = Dense(units = mp.hid_layer7, activation='relu')(z)

    decoded_red = Dense(units = mp.output_layer, activation='sigmoid')(x)
    decoded_green = Dense(units = mp.output_layer, activation='sigmoid')(y)
    decoded_blue = Dense(units = mp.output_layer, activation='sigmoid')(z)

    newModel = Model(encoding, [decoded_red, decoded_green, decoded_blue])

    # assign weights to decoder
    autoencoder = load_model(mp.LoadDir + 'autoencoder_sharedLatent.h5')
    newModel.layers[1].set_weights(autoencoder.layers[14].get_weights())
    newModel.layers[2].set_weights(autoencoder.layers[15].get_weights())
    newModel.layers[3].set_weights(autoencoder.layers[16].get_weights())
    newModel.layers[4].set_weights(autoencoder.layers[17].get_weights())
    newModel.layers[5].set_weights(autoencoder.layers[18].get_weights())
    newModel.layers[6].set_weights(autoencoder.layers[19].get_weights())
    newModel.layers[7].set_weights(autoencoder.layers[20].get_weights())
    newModel.layers[8].set_weights(autoencoder.layers[21].get_weights())
    newModel.layers[9].set_weights(autoencoder.layers[22].get_weights())
    newModel.layers[10].set_weights(autoencoder.layers[23].get_weights())
    newModel.layers[11].set_weights(autoencoder.layers[24].get_weights())
    newModel.layers[12].set_weights(autoencoder.layers[25].get_weights())

    return newModel


def define_decoder_nlp():
    encoding = Input(shape = (mp.hid_layer4,))

    # DECODER
    x = Dense(units = 192, activation = 'relu')(encoding)
    # x = BatchNormalization()(x)
    x = Dense(units = 768, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dense(units = 1728, activation='relu')(x)
    # x = BatchNormalization()(x)

    decoding = Dense(units = 3072, activation='sigmoid')(x)

    newModel = Model(encoding, decoding)
    autoencoder = load_model(mp.LoadDir + 'autoencoder_simple_nlp.h5')
    
    newModel.layers[1].set_weights(autoencoder.layers[5].get_weights())
    newModel.layers[2].set_weights(autoencoder.layers[6].get_weights())
    newModel.layers[3].set_weights(autoencoder.layers[7].get_weights())
    newModel.layers[4].set_weights(autoencoder.layers[8].get_weights())
    
    return newModel
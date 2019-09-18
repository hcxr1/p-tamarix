# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:26:09 2019

Model plotting utilities
Last Modified: 27/8/2019

@author: Rivan
"""
import matplotlib.pyplot as plt
import numpy as np
from .. import model_param as p



def display_model_evaluation(xtest_red, xtest_green, xtest_blue,
                             decoded_imgs_red, decoded_imgs_green, decoded_imgs_blue, counter = 0):
    """ 
        Show Reconstruction images
    """
    n = 10
    test_imgs = np.zeros(p.img_size)
    dec_imgs = np.zeros(p.img_size)
    
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        test_imgs[:,:,0] = xtest_red[counter + i].reshape(32, 32)
        test_imgs[:,:,1] = xtest_green[counter + i].reshape(32, 32)
        test_imgs[:,:,2] = xtest_blue[counter + i].reshape(32, 32)
        
        plt.imshow(test_imgs)
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display reconstruction
        ax = plt.subplot(2, n, i + n + 1)
        dec_imgs[:,:,0] = decoded_imgs_red[counter + i].reshape(32, 32)
        dec_imgs[:,:,1] = decoded_imgs_green[counter + i].reshape(32, 32)
        dec_imgs[:,:,2] = decoded_imgs_blue[counter + i].reshape(32, 32)
        
        plt.imshow(dec_imgs)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()



def display_history(history):
    """ list all data in history """
    
    print(history.history.keys())
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


   
def display_bottleneck(encoded_imgs, counter = 0):
    """Display the bottleneck layer"""
    n = 5
    plt.figure(figsize=(20, 8))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(encoded_imgs[counter + i])
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        
        ax = plt.subplot(2, n, i + n + 1)
        plt.hist(encoded_imgs[counter + i])
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
    plt.show()
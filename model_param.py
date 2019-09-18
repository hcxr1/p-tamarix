# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 06:50:57 2019

Last Modified: 28/8/2019

@author: Rivan
"""
import os


# Checkpoints Directory
LoadDir = "C:/Users/Rivan/Desktop/ptamarix/checkpoints/"

# Train Directory
trainDir = "C:/Users/Rivan/Desktop/Experiments/Train Images (Color)/"

# Test Directory
testDir1 = "C:/Users/Rivan/Desktop/Experiments/Test Images (Mixed Single)/"
testDir2 = "C:/Users/Rivan/Desktop/Experiments/Test Images (Mixed Double)/"
testDir3 = "C:/Users/Rivan/Desktop/Experiments/Test Videos/"

frame_size = img_size = (32,32,3)

display_size = (512,512,3)

input_layer = 1024
hid_layer1 = 576
hid_layer2 = 256
hid_layer3 = 64
hid_layer4 = bottleneck_layer = 10 # bottleneck layer
hid_layer5 = hid_layer3
hid_layer6 = hid_layer2
hid_layer7 = hid_layer1
output_layer = input_layer

learning_rate = 1e-5
loss_fx = 'mean_absolute_error' # Alternatively can use: 'mean_squared_error' 
                                # (performance wise: mean absolute error less affected by outliers)

batch_size = 128
epochs = 100
l1param = 3e-5
l2param = 10e-12


DEFAULT_SAMPLE_RATE = 44100


instruments_dict = { 'Black': 40,
                     'Red': 65,
                     'Green': 1,
                     'Blue': 13,
                   }


instruments_label = {
                     'Black': 0,
                     'Red': 1,
                     'Green': 2,
                     'Blue': 3,
                    }


instruments_label_rev = {
                     0: 'Black',
                     1: 'Red',
                     2: 'Green',
                     3: 'Blue',
                    }
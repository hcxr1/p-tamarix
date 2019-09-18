# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:25:16 2019

Visual to Audio Cross-Modal Conversion

Last Modified: 28/8/2019

@author: Rivan
"""

import numpy as np
from skimage.transform import resize
import model_param as p
from colorspace import rgb2hsl
from collections import Counter

  
def preprocess_frame(frame):
    """
        preprocess the video frame according to the autoencoder input
    """
    processed_frame = resize(frame, p.frame_size, anti_aliasing = True)
    
    processed_frame_red = processed_frame[:,:,2]
    processed_frame_green = processed_frame[:,:,1]
    processed_frame_blue = processed_frame[:,:,0]
    
    processed_frame_red = processed_frame_red.reshape(1, 1024)
    processed_frame_green = processed_frame_green.reshape(1, 1024)
    processed_frame_blue = processed_frame_blue.reshape(1, 1024)
        
    return processed_frame_red, processed_frame_green, processed_frame_blue



def preprocess_grayscale(frame):
    """ 
        Convert and preprocess frame as grayscale 
    """
    
    # Convert to grayscale
    def rgb2gray(x):
        #x has shape (width, height, channels)
        return (0.2989 * x[:,:,:1]) + (0.5870 * x[:,:,1:2]) + (0.1140 * x[:,:,-1:])

    processed_frame = resize(frame, p.frame_size, anti_aliasing = True)
    
    processed_frame_gray = rgb2gray(processed_frame)
    processed_frame_gray = processed_frame_gray.reshape(1,1024)
    
    return processed_frame_gray
    


def upscale_frame(image, display_size = p.display_size):
    """
        upscale the output image
    """
    return resize(image, p.display_size, anti_aliasing = True)



def color_define(hue, saturation, luminance):
    """
        Threshold the color value in HSL colorspace
    """
    if (luminance < 0.08): # 0.15
        color = 'Black'
    else:
        if (hue >= 0 and hue < 1/12.):
            color = 'Red'
        elif (hue >= 1/12. and hue < 1/6.):
            color = 'Red'# 'Orange'
        elif (hue >= 1/6. and hue < 1/3.):
            color = 'Green'# 'Yellow'
        elif (hue >= 1/3. and hue < 1/2.):
            color = 'Green'
        elif (hue >= 1/2. and hue < 2/3.):
            color = 'Blue'
        elif (hue >= 2/3. and hue < 5/6.):
            color = 'Blue'
        elif (hue >= 5/6. and hue < 1):
            color = 'Red'
    
    return color



def instrument_mapping_sL(decoder, bottleneck):
    """
        Map the music instruments used based on the color value of the bottlenecks
        (for model sharedLatent)
                
        Instruments:
            - Black => Violin (41)
            - Red => Alto Sax (65)
            - Green => Acoustic Grand Piano (1)
            - Blue => Xylophone (14)
            
        Ref: https://www.midi.org/specifications/item/gm-level-1-sound-set
    """

    instrument_list = np.zeros(p.hid_layer4, dtype = int)
    
    print('bottleneck: ',bottleneck)
    
    temp_bottleneck = np.zeros(p.hid_layer4)
    temp_bottleneck = bottleneck
    temp_bottleneck = temp_bottleneck.reshape(1, p.hid_layer4)
    decoded_r, decoded_g, decoded_b = decoder.predict(temp_bottleneck)

    r = decoded_r.mean()
    g = decoded_g.mean()
    b = decoded_b.mean()

    h, s, l = rgb2hsl(r,g,b)
    color = color_define(h,s,l)
    
    instrument_list[:] = p.instruments_dict[color]
    print('Color ',color)
    
    return instrument_list



def instrument_mapping_nlp(decoder,bottleneck):
    """
        Map the music instruments used based on the color value of the bottlenecks
        (for model simple_nlp)
                
        Instruments:
            - Black => Violin (41)
            - Red => Alto Sax (65)
            - Green => Acoustic Grand Piano (1)
            - Blue => Xylophone (14)
            
        Ref: https://www.midi.org/specifications/item/gm-level-1-sound-set
    """
    instrument_list = np.zeros(p.hid_layer4, dtype = int)
    
    print('bottleneck: ',bottleneck)
    
    temp_bottleneck = np.zeros(p.hid_layer4)
    temp_bottleneck = bottleneck
    temp_bottleneck = temp_bottleneck.reshape(1, p.hid_layer4)
    decoded = decoder.predict(temp_bottleneck)

    r = decoded[0:1024].mean()
    g = decoded[1024:2048].mean()
    b = decoded[2048:3072].mean()

    h, s, l = rgb2hsl(r,g,b)
    color = color_define(h,s,l)
    
    instrument_list[:] = p.instruments_dict[color]
    print('Color ',color)
    
    return instrument_list



def isDrum_mapping(instrument_list):
    """
        Return a list of boolean of repercussive instruments labelling
        (Used only with magenta libraries)
    """        
    isDrums = []
    # If Percussive switch
    for i in range(len(instrument_list)):
        if instrument_list[i] > 112:
            isDrums.append(True)
        else:
            isDrums.append(False)
    
    isDrum_list = np.asarray(isDrums)
    return isDrum_list
    
 
 
def velocity_mapping():
    """
        Return an array of Velocity parameter
        (Used in sharedLatent and simple_nlp)
    """

    velocity_list = np.ones(p.hid_layer4, dtype = int) * 64
    
    return velocity_list



def channel_mapping():
    """
        Return an array of MIDI channel value
    """
    channel_list = np.arange(10)
    
    return channel_list



def instrument_mapping_fc(botRed, botGreen, botBlue):
    """
        Map the music instruments used based on the color value of the bottlenecks
        (for model independent fully connected networks)
                
        Instruments:
            - Black => Violin (41)*
            - Red => Alto Sax (65)*
            - Green => Bright Acoustic Piano (2)*
            - Blue => Xylophone (14)*
            
        Ref: https://www.midi.org/specifications/item/gm-level-1-sound-set
    """
    instrument_list = np.zeros(p.hid_layer4, dtype = int)
    # velocity_list = np.zeros(p.bottleneck_layer, dtype = int)
    labels = np.zeros(p.hid_layer4, dtype = int)
    
    maxVal = np.max([botRed, botGreen, botBlue])
    r = botRed/maxVal
    g = botGreen/maxVal
    b = botBlue/maxVal

    for i in range(p.hid_layer4):
        h, s, l = rgb2hsl(r[i],g[i],b[i])
        # velocity_list[i] = l * 127
        color = color_define(h,s,l)   
        labels[i] = p.instruments_label[color]

    print('instrument labels: ',labels)     
    countLabel = Counter(labels)   
    countVal = np.zeros(p.hid_layer4, dtype = int)
    
    for i in range(p.bottleneck_layer):
        countVal[i] = countLabel[i]
    
    # Extract top 2 Colors (Inefficient implementation, but works)
    indMax = np.zeros(2, dtype = int)
    cc = 0
    for i in range(p.bottleneck_layer):
        if (countVal[i] == np.max(countVal)):
            indMax[cc] = i
            cc = cc + 1
            break
    
    countVal[indMax[cc-1]] = 0       
    for i in range(p.bottleneck_layer):
        if (countVal[i] == np.max(countVal)):
            indMax[cc] = i
            cc = cc + 1
            break
    print(indMax)
    
    for i in range(p.bottleneck_layer):
        if labels[i] == indMax[0]:
            instrument_list[i] = p.instruments_dict[p.instruments_label_rev[indMax[0]]]
            print('color',p.instruments_label_rev[indMax[0]])
        elif labels[i] == indMax[1]:
            instrument_list[i] = p.instruments_dict[p.instruments_label_rev[indMax[1]]]
            print('color',p.instruments_label_rev[indMax[1]])
        else:
            instrument_list[i] = p.instruments_dict[p.instruments_label_rev[indMax[0]]]
            print('color',p.instruments_label_rev[indMax[0]])

    
    return instrument_list


# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:53:08 2019

Display the frame and the reconstructed results

Last Modified: 24/8/2019

@author: Rivan
"""

import numpy as np
import cv2
from conversion import upscale_frame
import model_param as p

def display_output(inFrame, outRed, outGreen, outBlue):
    """
        Display reconstruction output frame and input frame side by side
        Input -> inFrame: original frame taken
                 outRed: predicted Red Layer
                 outGreen: predicted Green Layer
                 outBlue: predicted Blue Layer
                 
        Output in BGR colorspace
    """
    
    outFrame = np.zeros(p.frame_size)
   
    # Display the frame
    temp1 = upscale_frame(inFrame)
    
    outFrame[:,:,0] = outBlue.reshape(32,32)
    outFrame[:,:,1] = outGreen.reshape(32,32)
    outFrame[:,:,2] = outRed.reshape(32,32)
    temp2 = upscale_frame(outFrame)
    
    merged = np.concatenate((temp1, temp2), axis = 1)
    cv2.imshow('frame', merged)   

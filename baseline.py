# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 06:56:57 2019

Model: Baseline 2
Last Modifited: 1/9/2019

@author: Rivan
"""

# import keras
from keras.models import load_model, Model
from conversion import preprocess_frame
from conversion import channel_mapping
from conversion import instrument_mapping_fc, velocity_mapping
from display import display_output

import numpy as np
import cv2
import time
import model_param as p
import mido

# print(keras.__version__)
# print(mido.__version__)

sample_rate = p.DEFAULT_SAMPLE_RATE


if __name__ == '__main__':
    
    model_red = load_model(p.LoadDir + 'autoencoder_baseline_red.h5')  
    model_green = load_model(p.LoadDir + 'autoencoder_baseline_green.h5')  
    model_blue = load_model(p.LoadDir + 'autoencoder_baseline_blue.h5')  
    # model_gray = load_model(p.LoadDir + 'autoencoder_grayscale.h5')
               
    # Extract the bottleneck layers
    bottleneck_model_red = Model(inputs = model_red.input, outputs = model_red.get_layer(index = 4).output)
    bottleneck_model_green = Model(inputs = model_green.input, outputs = model_green.get_layer(index = 4).output)
    bottleneck_model_blue = Model(inputs = model_blue.input, outputs = model_blue.get_layer(index = 4).output)
    # bottleneck_model_gray = Model(inputs = model_gray.input, outputs = model_gray.get_layer(index = 4).output) 
    
    print("Bottleneck_model successfully extracted!\n")

    # 0 = Train, 1 = Test Images (Mixed Single), 2 = Test Images (Mixed Double), 3 = Test Video
    session = int(input("Session: ")) 
    if (session == 3):
        filename = input('Video Filename: ')
        cap = cv2.VideoCapture(p.testDir3 + filename +'.mp4')
    else:
        cap = cv2.VideoCapture(0) # Webcam
        
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer. Uncomment below to save output to file
    """
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
 
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    """
    # Get audio port    
    print(mido.get_output_names())
    port = mido.open_output()
    
    prevBottleneck = np.zeros(10)
    prevBottleneck = prevBottleneck.astype('int')
    firstTime = True


    channels = channel_mapping()
    arrvelocity = velocity_mapping()
    
    
    while (True):
        start_time = time.time()
        
        # Read the frames      
        if (session < 3):
            ### This Part is for Training Session and Test Images
            imgFile = input("Input Image: ")
            if (imgFile == 'done'):
                break
            
            if (session == 0):
                frame = cv2.imread(p.trainDir + imgFile +'.png')
            elif (session == 1):
                frame = cv2.imread(p.testDir1 + imgFile +'.png')
            elif (session == 2):
                frame = cv2.imread(p.testDir2 + imgFile + '.png')

            ret = True
        else:
            ret, frame = cap.read()
        
        
        outFrame = np.zeros(p.frame_size)
        
        if ret == True:
            # Write the frame into the file 'output.avi'
            # if (MODE == 1):
            #    out.write(frame)
            
            # Preprocess the frame
            frame_red, frame_green, frame_blue = preprocess_frame(frame)
        
            # Evaluate frame on the Model
            botRed = bottleneck_model_red.predict(frame_red)
            botGreen = bottleneck_model_green.predict(frame_green)
            botBlue = bottleneck_model_blue.predict(frame_blue)
            
            outRed = model_red.predict(frame_red)
            outGreen = model_green.predict(frame_green)
            outBlue = model_blue.predict(frame_blue)
            
            
            # Display the frame
            display_output(frame, outRed, outGreen, outBlue)
            
            
            # Bottlenecks to Grayscale as pitches
            botRed = botRed.astype(int)
            botRed = botRed.reshape(p.bottleneck_layer)
            botGreen = botGreen.astype(int)
            botGreen = botGreen.reshape(p.bottleneck_layer)
            botBlue = botBlue.astype(int)
            botBlue = botBlue.reshape(p.bottleneck_layer)
                      
            avgBottleneck =  0.2989 * botRed + 0.5870 * botGreen + 0.1140 * botBlue # bottleneck_model_gray.predict(preprocess_grayscale(frame))
            
            # Processing bottlenecks
            bottleneck = np.round(avgBottleneck) # * 1.5 + 20
            bottleneck = np.clip(bottleneck.astype(int) * 2 + 20, a_min = 0, a_max = 127)
            bottleneck = bottleneck.reshape(p.hid_layer4)
            
            # Instruments mapping
            music_instruments = instrument_mapping_fc(botRed, botGreen, botBlue)
            
            # print('r', botRed)
            # print('g', botGreen)
            # print('b', botBlue)
            print('pitch', bottleneck)
            print('velocity',arrvelocity)
            
            # print('t', time.time()-start_time)
                
            checkBottleneck = bottleneck != prevBottleneck # Check bottleneck for any change in keypress
                
            for i in range(p.hid_layer4):
                msg = mido.Message('program_change',channel = channels[i], program = music_instruments[i])
                port.send(msg)
                
            for i in range(p.hid_layer4):
                if (checkBottleneck[i] and not(firstTime)):
                    msg = mido.Message('note_off', channel = channels[i], note = prevBottleneck[i],
                                       velocity = arrvelocity[i], time = 0)
                    port.send(msg)
                    msg = mido.Message('note_on', channel = channels[i], note = bottleneck[i],
                                       velocity = arrvelocity[i], time = 0)
                    port.send(msg)
                elif (firstTime):
                    msg = mido.Message('note_on', channel = channels[i], note = bottleneck[i],
                                       velocity = arrvelocity[i], time = 0)
                    port.send(msg)
                
            prevBottleneck = bottleneck
            firstTime = False
               
            
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        else:
            break
    
    # When everything done, release the video capture and video write objects
    cap.release()
    port.close()
    # out.release()
    cv2.destroyAllWindows() # Closes all the frames
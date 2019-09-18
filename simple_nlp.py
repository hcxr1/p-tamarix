# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 06:41:06 2019

Model: simple nlp fully connected autoencoder
                Set a musical instrument to play the output of the nodes
                    - Violin
                    - AltoSax
                    - Grand Acoustic Piano
                    - Xylophone   
Last Modified: 30/8/2019

@author: Rivan
"""

# import keras
from keras.models import load_model, Model
from conversion import preprocess_frame, channel_mapping
from conversion import velocity_mapping, instrument_mapping_nlp
from models.latent_decoder import define_decoder_nlp
import numpy as np
import cv2
import time
import model_param as mp
import mido
from display import display_output


# print(keras.__version__)
# print(mido.__version__)


sample_rate = mp.DEFAULT_SAMPLE_RATE

if __name__ == '__main__':
    
    # Load Model
    model = load_model(mp.LoadDir + 'autoencoder_simple_nlp.h5')   
    # model.summary()
    
    print('This is the encoder model\n')
    # Extract the bottleneck layers
    bottleneck_model = Model(inputs = model.input, outputs = model.get_layer(index = 4).output)
    bottleneck_model.summary()
    
    decoder = define_decoder_nlp()
    print("Model successfully extracted!\n")

    # 0 = Train, 1 = Test Images (Mixed Single), 2 = Test Images (Mixed Double), 3 = Test Video
    session = int(input("Session: ")) 
    if (session == 3):
        filename = input('Video Filename: ')
        cap = cv2.VideoCapture(mp.testDir3 + filename +'.mp4')
    else:
        cap = cv2.VideoCapture(0) # Webcam
        
    
    #   Default resolutions of the frame are obtained.The default resolutions are system dependent.
    #   We convert the resolutions from float to integer. Uncomment below to save output to file
    """
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
 
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
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
                frame = cv2.imread(mp.trainDir + imgFile +'.png')
            elif (session == 1):
                frame = cv2.imread(mp.testDir1 + imgFile +'.png')
            elif (session == 2):
                frame = cv2.imread(mp.testDir2 + imgFile + '.png')
            
            ret = True
        else:
            ret, frame = cap.read()

         
        if ret == True:
            # Write the frame into the file 'output.avi'
            # if (MODE == 1):
            #    out.write(frame)
            
            # Preprocess the frame
            frame_red, frame_green, frame_blue = preprocess_frame(frame)            
            frame_ct = np.concatenate((frame_red, frame_green, frame_blue), axis=None)
            frame_ct = frame_ct.reshape(1,3072)
            
            # Evaluate frame on the Model
            bottleneck = bottleneck_model.predict(frame_ct)
            
            outFrame = model.predict(frame_ct)
            
            display_output(frame, outFrame[:,0:1024], outFrame[:,1024:2048], outFrame[:,2048:3072])
            
            
            # Mapping of the musical parameters
            bottleneck = bottleneck.reshape(mp.bottleneck_layer)
            bottleneck = np.round(bottleneck)
            bottleneck = bottleneck.astype(int)
            music_instruments = instrument_mapping_nlp(decoder,bottleneck) 
            
            # print('t: ',time.time()-start_time)
            for i in range(mp.hid_layer4):
                msg = mido.Message('program_change',channel = channels[i], program = music_instruments[i])
                port.send(msg)
                
            checkBottleneck = bottleneck != prevBottleneck # Check bottleneck for any change in keypress
                
            for i in range(mp.hid_layer4):
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
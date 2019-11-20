#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:00:14 2019

@author: jayeshmehta
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import cv2
import os
from PIL import Image


def new_user():

    cam = cv2.VideoCapture(0)
    cam.set(3, 720) # set video width
    cam.set(4, 720) # set video height
    
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    
    # For each person, enter one numeric face id
    face_id = input('\n enter Employee id end press Enter ==>  ')
    # Get names for each person
    name_id = input('\n enter Employee Name end press Enter ==>  ')
    
    
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.5, 5)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
    
            # Save the captured image into the datasets folder
            cv2.imwrite("Dataset/User." + str(name_id)+ '-' + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
    
            cv2.imshow('image', img)
    
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 75: # Take 75 face sample and stop video
             break
    
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    


    
def train_model():
    # Path for face image database
    path = 'dataset'
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    
    # function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids= []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1].split('-')[1])
            name = str(os.path.split(imagePath)[-1].split("-")[0].split('.')[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    
    # Save the model into trainer/trainer.yml
    #recognizer.write('trainer/trainer.yml') 
    recognizer.save('trainer/trainer.yml')
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    
    
    
def detect():
 
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #iniciate id counter
    id = 0
    
    # names related to ids: example ==> Marcelo: id=1,  etc
    path = 'Dataset'
    
    #Function to get names from images
    '''def getNames(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        names= []
        for imagePath in imagePaths:
            name = str(os.path.split(imagePath)[-1].split("-")[0].split('.')[1])
            names.append(name)
        return names
    names1 = getNames(path)
    
    for name in names1:
        if name not in names:
            names.append(name)'''
    
    
    
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 720) # set video widht
    cam.set(4, 720) # set video height
    
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    
    while True:
        ret, img =cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.5,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )
    
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
    
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('camera',img) 
    
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
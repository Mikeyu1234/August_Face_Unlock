import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import os
import sys

camera = PiCamera()
camera.resolution =(1024,768)
camera.framerate = 30

rawCapture = PiRGBArray(camera, camera.resolution)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

name = input('What is her name?')
dirName = './images/'+name
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print('Dir Created')
else:
    print('name already exists')
    sys.exit()


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame.array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    count = 0
    for (x, y, w, h) in faces:
        roiGray = gray[y:y+h, x:x+w]
        fileName = dirName + "/" + name + str(count) + ".jpg"
        cv2.imwrite(fileName, roiGray)
        cv2.imshow("face", roiGray)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count += 1
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    rawCapture.truncate(0)

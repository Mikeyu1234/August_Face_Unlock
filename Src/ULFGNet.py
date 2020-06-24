import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import os
import sys

import argparse
import time
from math import ceil
#import onnx
#import vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend


from cv2 import dnn

camera = PiCamera()
camera.resolution =(640,480)
camera.framerate = 30

rawCapture = PiRGBArray(camera, camera.resolution)


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

name = input('What is her name?')
dirName = '../images/'+name
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print('Dir Created')
else:
    print('name already exists')
    sys.exit()


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame.array
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_mean = np.array([127,127,127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)


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

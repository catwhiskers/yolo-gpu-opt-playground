#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os 
from os import listdir
from os.path import isfile, join
import time

# prepare config files
labelsPath = os.path.sep.join([".","coco.names"])
weightsPath = os.path.sep.join([".", "yolov3.weights"])
configPath = os.path.sep.join([".", "yolov3.cfg"])

# build yolo network
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
image_dir = "images"
image_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
timeStr = time.time()


with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
def get_next_image_batch(image_files, start, batch_size):
    imgs = []
    img_meta = []
    next_start = start + batch_size
    for img_name in image_files[start:min(start+batch_size, len(image_files))]:
        img = cv2.imread(join(image_dir, img_name))
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
        img_meta.append((height, width, channels))
        imgs.append(img)
    if len(imgs) < batch_size:
        next_start = -1
    return imgs, img_meta, next_start

start = 0
batch_size = 8

while True:
    imgs, img_meta, next_start = get_next_image_batch(image_files, start, batch_size)
    blob = cv2.dnn.blobFromImages(imgs, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for i, layeroutput in enumerate(outs):
        height, width, channels = img_meta[i]
        for out in layeroutput:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(len(indexes))
    if next_start == -1: 
        break







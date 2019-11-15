from ctypes import *
from collections import namedtuple
import math
import random
import os
import cv2
import numpy as np
import time
import darknet.core as darknet

detected_objects = []
Vect2 = namedtuple('Vect2', 'x y')

class yolo_obj(object):
    def __init__(self, d):
        self.__dict__ = d


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def get_detection_data(detection):
    label = detection[0] 
    precision = detection[1] 
    box = detection[2]
    x, y = box[0], box[1]
    w, h = box[2], box[3]
    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
    obj = yolo_obj({
            'label': detection[0], 
            'precision':detection[1], 
            'pos':Vect2(xmin, ymin),
            'size':Vect2(w, h),
            'center':Vect2(x, y)
    })
    return obj

def cvDrawBox(obj, img):
    pt1 = (int(obj.pos.x), int(obj.pos.y))
    pt2 = (int(obj.pos.x+obj.size.x), int(obj.pos.y+obj.size.y))

    color = (0, 255, 0)
    if obj.precision < 0.6:
        color = (255, 255, 0)
    elif obj.precision < 0.4:
        color = (255, 153, 51)
    elif obj.precision < 0.2:
        color = (128, 0, 0)

    cv2.rectangle(img, pt1, pt2, color, 1)
    cv2.putText(img,
                obj.label.decode() +
                " [" + str(round(obj.precision * 100, 2)) + "]",
                (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO(cfgPath=None, wgtPath=None, mtPath=None):

    global metaMain, netMain, altNames, detected_objects

    if(cfgPath and wgtPath and mtPath):
        configPath = cfgPath
        weightPath = wgtPath
        metaPath = mtPath
    else:
        configPath = "./cfg/yolov3-tiny.cfg"
        weightPath = "./weights/yolov3-tiny.weights"
        metaPath = "./cfg/coco.data"
    

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("test.mp4")
    if( not cap.isOpened()) :
        print("Unable to open camera")
        return -1;
    #cap.set(3, 800)
    #cap.set(4, 800)
    #out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 5.5, (int(cap.get(3)*2), int(cap.get(4)*2)) )
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain), darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        # we collect YOLO detections
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        image = frame_resized

        # parsing detections and drawing boxes
        for detection in detections:
            obj = get_detection_data(detection)
            image = cvDrawBox(obj, image)
            print(obj.label, obj.center, obj.size)
            detected_objects.append(obj)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, ( int(cap.get(3)*2), int(cap.get(4)*2) ), interpolation=cv2.INTER_LINEAR)
        

        # display FPS text
        cv2.putText(image, str(round(1.0/(time.time()-prev_time), 2))+" FPS",
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255, 255, 255], 2, bottomLeftOrigin=False)
        
        cv2.imshow('YOLO', image)
        # out.write(image)

        print("----------")

        detected_objects = []

        key = cv2.waitKey(33)
        if key == 27 : break

    cap.release()
    #out.release()

if __name__ == "__main__":
    YOLO()

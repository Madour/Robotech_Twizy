from ctypes import *
from collections import namedtuple
import math
import random
import os
import cv2
import numpy as np
import time
import darknet.core as darknet

Vect2 = namedtuple('Vect2', 'x y')

class yolo_obj(object):
    def __init__(self, d):
        self.__dict__ = d

    def print_info(self):
        print(
            round(self.precision,3)*100,
            "|",self.label,
            "| pos:", (round(self.pos.x, 2), round(self.pos.y, 2)),
            "| size:", (round(self.size.x, 2), round(self.size.y, 2))
        )


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def get_detection_data(detection):
    list = []
    label = detection[0] 
    precision = detection[1] 
    box = detection[2]
    x, y = box[0], box[1]
    w, h = box[2], box[3]
    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
    obj = yolo_obj({
            'label': detection[0].decode(), 
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
                obj.label +
                " [" + str(round(obj.precision * 100, 2)) + "]",
                (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1)
    return img


class YOLO(object):

    def __init__(self, cfgPath=None, wgtPath=None, mtPath=None, autoload=False, autostart=False):
        self.metaMain = None
        self.netMain = None
        self.altNames = None
        self.detected_objects = []

        self.danger_objects = ["person", "cat", "dog", "car", "bicycle", "motorbike", "bus", "train", "truck"]

        if(cfgPath and wgtPath and mtPath):
            self.configPath = cfgPath
            self.weightPath = wgtPath
            self.metaPath = mtPath
        else:
            self.configPath = "./cfg/yolov3-tiny.cfg"
            self.weightPath = "./weights/yolov3-tiny.weights"
            self.metaPath = "./cfg/coco.data"
    

        # check for config files
        if not os.path.exists(self.configPath):
            raise ValueError("Invalid config path `" + os.path.abspath(self.configPath)+"`")
        if not os.path.exists(self.weightPath):
            raise ValueError("Invalid weight path `" + os.path.abspath(self.weightPath)+"`")
        if not os.path.exists(self.metaPath):
            raise ValueError("Invalid data file path `" + os.path.abspath(self.metaPath)+"`")
        
        if autostart:
            self.load()
            self.start()
        elif autoload:
            self.load()
        else:
            pass

        self.camera_size = Vect2(0, 0)

    def load(self):
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(self.configPath.encode("ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(self.metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(self.metaPath) as metaFH:
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
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass
    

    def start(self, display_window=False):
        cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture("test.mp4")
        if( not cap.isOpened()) :
            print("Unable to open camera")
            return -1;
        self.camera_size = Vect2(int(cap.get(3)),  int(cap.get(4)))
        #cap.set(3, 800)
        #cap.set(4, 800)
        #out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 5.5, (int(cap.get(3)*2), int(cap.get(4)*2)) )
        print("Starting the YOLO loop...")
        # Create an image we reuse for each detect
        darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                    darknet.network_height(self.netMain),3)
        while True:
            prev_time = time.time()
            ret, frame_read = cap.read()
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (darknet.network_width(self.netMain), darknet.network_height(self.netMain)),
                                       interpolation=cv2.INTER_LINEAR)
            image = frame_resized
            
            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    
            # collect YOLO detections
            detections = darknet.detect_image(self.netMain, self.metaMain, darknet_image, thresh=0.25)
            
            for detection in detections:
                obj = get_detection_data(detection)
                self.detected_objects.append(obj)
                #obj.print_info()
                if display_window: image = cvDrawBox(obj, image)
            
            # analyze collected detections
            self.process()
            
            self.detected_objects = []
            
            if display_window:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, ( self.camera_size.x*2, self.camera_size.y*2 ), interpolation=cv2.INTER_LINEAR)
                
                # display FPS text
                cv2.putText(image, str(round(1.0/(time.time()-prev_time), 2))+" FPS",
                            (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255, 255, 255], 2, bottomLeftOrigin=False)
                
                cv2.imshow('YOLO', image)
                #out.write(image)
    
                key = cv2.waitKey(1)
                if key == 27 : break # escape key pressed, we quit
    
        if display_window: cap.release()
        #out.release()

    def process(self):
        # write here how to analyze and interpret detected_objects
        # hint: we need to get the real size of detected objects
        danger = None

        for detected_object in self.detected_objects:
            # do some stuff with detected_object
            #example: 
            if detected_object.label in self.danger_objects and detected_object.size.x*100/darknet.network_width(self.netMain) >= 40:
                danger = detected_object.label

        if danger:
            print("/!\ DANGER /!\ ", danger, "ON THE ROAD")
        else:
            print("All safe !")



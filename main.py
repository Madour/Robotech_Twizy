
from dll.darknet_video import YOLO

YOLO(cfgPath = "./cfg/yolov3.cfg",
     wgtPath = "./weights/yolov3.weights",
      mtPath = "./cfg/coco.data"
    )
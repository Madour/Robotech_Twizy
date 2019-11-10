from dll.darknet_video import YOLO

# détection avec YOLO à partir de la webcam
YOLO(cfgPath = "./cfg/yolov3.cfg",
     wgtPath = "./weights/yolov3.weights",
      mtPath = "./cfg/coco.data"
    )
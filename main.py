import darknet

# détection avec YOLO à partir de la webcam

# initialisation
detector = darknet.yolo.YOLO(
			cfgPath = "./cfg/yolov3.cfg",
            wgtPath = "./weights/yolov3.weights",
            mtPath  = "./cfg/coco.data"
            )

detector = darknet.yolo.YOLO()

# loading the network
detector.load()

# starting webcam detection
detector.start(display_window=True)
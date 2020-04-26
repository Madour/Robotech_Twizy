import darknet.yolo as dn

# détection avec YOLO à partir de la webcam

# initialisation
detector = dn.YOLO(
			cfgFile = "./cfg/yolov3.cfg",
            wgtFile = "./weights/yolov3.weights",
            mtFile  = "./cfg/coco.data"
            )

# loading the network
detector.load()

# starting webcam detection
detector.start(display_window=True)
# Robotech: projet TWIZY !

### Description

Les données des objets détectés par YOLO se présentent sous la forme `(label, precision, (x_center, y_center, width, height))`


### How to run

Nécessite d'avoir compilé et généré les librairies dynamiques de darknet.

Placer les fichiers .weights de YOLOv3 (yolov3.weights) dans le dossier weights.

##### Pour Linux :
Placer libdarknet.so à la racine du projet (à coté du main.py)

##### Pour Windows :
Placer les DLL nécessaires (yolo_cpp_dll.dll, pthreadVC2.dll et opencv_videoio_ffmpeg412_64.dll) dans le dossier darknet.

Lancer la commande `python3 main.py`

Si les performances sont trop faibles (moins de 5 FPS) , il est possible d'utiliser le modèle plus léger yolov3-tiny.weights.
Ajouter ce fichier dans le dossier weights, puis dans le `main.py` faire les modifications nécessaires:
Changer : `cfgFile = "./cfg/yolov3.cfg", wgtPath = "./weights/yolov3.weights",`
en :      `cfgFile = "./cfg/yolov3-tiny.cfg", wgtPath = "./weights/yolov3-tiny.weights",`

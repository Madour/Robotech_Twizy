# TWIZY YOLO !

### Description

Ce dépot contiendra les programmes qui ont pour objectif d'utiliser YOLO
pour la reconnaissance des objets en temps réelle à partir de la webcam.

Il faudra ensuite traiter les données de YOLO qui sont en fait des tuple
décrivant les bounding box des objets détectés.
Ces données se présentent sous la forme `(label, precision, (x_center, y_center, width, height))`


### How to run

Toute les dlls sont fournis. Il faudra juste télecharger le fichier 
yolov3.weights à partir du site de darknet/yolo et le placer dans le
dossier "weights".
Pour un test rapide, ouvrez le fichier "main.py" avec `python main.py` (nécessite python 3)
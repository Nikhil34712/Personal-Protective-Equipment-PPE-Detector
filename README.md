# Personal-Protective-Equipment-PPE-Detector

This project implements Personal Protective Equipment (PPE) detection using YOLOv5(PyTorch)

The dataset can be downloaded using Roboflow:

from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key")
project = rf.workspace("personal-protective-equipment").project("ppes-kaxsi")
dataset = project.version(8).download("yolov5")

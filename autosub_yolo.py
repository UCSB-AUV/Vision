# -*- coding: utf-8 -*-
"""AutoSub Yolo

Trained on old data for one specific task (torpedo)

Original file is located at
    https://colab.research.google.com/drive/1pWsY0Xriq6WHmesaV8cBf5H22l48VdRO
"""

!pip install ultralytics
!pip install roboflow

from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="KmhM0hbObRBKuJs68xKV")
project = rf.workspace("dummy-kmudt").project("robosub-r1r3p")
version = project.version(1)
dataset = version.download("yolov8")


model = YOLO("yolov8n.pt")

# Other options:
# yolov8s.pt
# yolov8m.pt
# yolov8l.pt

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,      #GPU
    workers=8,
    patience=20,
    project="robosub_detection",
    name="underwater_yolo"
)

from ultralytics import YOLO

model = YOLO("/best.pt")

results = model("/content/Robosub-1/train/images/Aayush_imgs_torpedo_253_jpg.rf.c106c474dc1d954a67969c4baccd3e69.jpg", save=True, conf=0.4)

print(results)

import cv2
import matplotlib.pyplot as plt
import os

image_path = "/content/Robosub-1/train/images/Aayush_imgs_torpedo_253_jpg.rf.c106c474dc1d954a67969c4baccd3e69.jpg"

label_path = image_path.replace("/images/", "/labels/")
label_path = os.path.splitext(label_path)[0] + ".txt"

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, _ = img.shape

with open(label_path, "r") as f:
    lines = f.readlines()

for line in lines:
    cls, x, y, bw, bh = map(float, line.split())

    x1 = int((x - bw/2) * w)
    y1 = int((y - bh/2) * h)
    x2 = int((x + bw/2) * w)
    y2 = int((y + bh/2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis("off")
plt.show()

results = model(image_path)

boxes = results[0].boxes

for box in boxes:
    cls_id = int(box.cls[0])

    confidence = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0]

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    print("Class:", model.names[cls_id])
    print("Confidence:", confidence)
    print("Center:", center_x, center_y)
    print("---")

from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source=0,      # webcam
    show=True,     # opens live window
    conf=0.3
)

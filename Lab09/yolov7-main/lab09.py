import numpy as np
from numpy import random
import cv2
import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import  plot_one_box

WEIGHT = './runs/train/yolov7-lab09/weights/best.pt'
# WEIGHT = './runs/train/yolov7-lab09/weights/last.pt'
# WEIGHT = './yolov7-tiny.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"

model = attempt_load(WEIGHT, map_location=device)
if device == "cuda":
    model = model.half().to(device)
else:
    model = model.float().to(device)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

path = '../lab09_test.mp4'
output_path = '../lab09_output.mp4'
cap = cv2.VideoCapture(path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码器
out = cv2.VideoWriter(output_path, fourcc, 33, (1280, 720))
while True:
    ret, image = cap.read()
    if not ret: 
        break
    
    image_orig = image.copy()
    image = letterbox(image, (640, 640), stride=64, auto=True)[0]
    if device == "cuda":
        image = transforms.ToTensor()(image).to(device).half().unsqueeze(0)
    else:
        image = transforms.ToTensor()(image).to(device).float().unsqueeze(0)

    with torch.no_grad():
        output = model(image)[0]
    output = non_max_suppression_kpt(output, conf_thres=0.25, iou_thres=0.65)[0]
    ## Return: list of detections, on (n,6) tensor per image [xyxy, confidance, class]
    
    ## Draw label and confidence on the image
    output[:, :4] = scale_coords(image.shape[2:], output[:, :4], image_orig.shape).round()
    for *xyxy, conf, cls in output:
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, image_orig, label=label, color=colors[int(cls)], line_thickness=1)
        
    # cv2.imshow("Detected", image_orig)
    out.write(image_orig)
    # if cv2.waitKey(30) & 0xFF == ord('q'):
    #     break
    # cv2.waitKey(1)

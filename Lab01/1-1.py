import cv2
import numpy as np

img = cv2.imread('images/nctu_flag.jpg')
x, y, _ = img.shape
for i in range(x):
    for j in range(y):
        if img[i, j][0] > 70 and img[i, j][0]*0.8 > img[i, j][1] and img[i, j][0]*0.8 > img[i, j][2]:
            continue
        else:
            new = sum(img[i, j])/3
            img[i, j] = [new, new, new]

cv2.imwrite('./output/output1-1.png', img)
cv2.imshow('img',img)
cv2.waitKey(0)
import cv2
import numpy as np

img = cv2.imread('./images/test.jpg')
x, y, _ = img.shape
new_img = np.zeros([x*3, y*3, 3], dtype=int)
for i in range(x*3):
    for j in range(y*3):
        a = round(i/3)
        if round(i/3) >= x:
            a = round(i/3) - 1
        b = round(j/3)
        if round(j/3) >= y:
            b = round(j/3) - 1
        
        new_img[i, j] = img[a, b]
cv2.imwrite('./output/2.png', new_img)
cv2.imshow('new_img',new_img)
cv2.waitKey(0)
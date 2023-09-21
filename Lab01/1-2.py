import cv2
import numpy as np

img = cv2.imread('./images/nctu_flag.jpg')
img = np.array(img, dtype=np.int32)
x, y, _ = img.shape
for i in range(x):
    for j in range(y):
        if (img[i, j][0] + img[i, j][1])*0.3 > img[i, j][2]:
            contrast = 100
            brightness = 40
            img[i, j] = [(c-127)*(contrast/127+1)+127+brightness for c in img[i, j]]

np.clip(img, 0, 255, out=img)
img = np.array(img, dtype=np.uint8)
cv2.imwrite('./output/1-2.png', img)
cv2.imshow('img',img)
cv2.waitKey(0)
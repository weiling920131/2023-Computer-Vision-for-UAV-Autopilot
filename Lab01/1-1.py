import cv2
import numpy as np

img = cv2.imread('images/nctu_flag.jpg')
new_img = img.copy()
h, w, _ = img.shape

for i in range(h):
    for j in range(w):
        B, G, R = img[i, j]
        if not (B > 70 and B * 0.8 > G and B * 0.8 > R):
            avg = B / 3 + G / 3 + R / 3
            new_img[i, j] = avg

cv2.imshow('1-1', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output/1-1.jpg', new_img)
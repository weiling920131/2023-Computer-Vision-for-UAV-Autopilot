import cv2
import numpy as np

img = cv2.imread('images/filtering.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5,5), 0)

x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
y_kernel = x_kernel.T
gx = np.array(cv2.filter2D(img, -1, x_kernel), dtype=np.int32)
gy = np.array(cv2.filter2D(img, -1, y_kernel), dtype=np.int32)

new_img = np.array((abs(gx) + abs(gy)), dtype=np.int32)
np.clip(new_img, 0, 255, out=new_img)
new_img = np.array(new_img, dtype=np.uint8)

cv2.imwrite('output/1.png', new_img)
cv2.imshow('1.png', new_img)
cv2.waitKey(0)

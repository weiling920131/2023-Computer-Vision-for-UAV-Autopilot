import cv2
import numpy as np

img = cv2.imread('images/1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)
h, w = img.shape

x = np.array([[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]])
y = np.array([[1, 2, 1],
              [0, 0, 0],
              [-1, -2, -1]])

gx = np.array(cv2.filter2D(img, -1, x), dtype=np.int32)
gy = np.array(cv2.filter2D(img, -1, y), dtype=np.int32)

new_img = np.array(np.sqrt(np.square(gx) + np.square(gy)), dtype=np.int32)

new_img = np.clip(new_img, 0, 255)
new_img = np.array(new_img, dtype=np.uint8)

cv2.imshow('1', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output/1.jpg', new_img)
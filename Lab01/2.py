import cv2
import numpy as np

img = cv2.imread('images/test.jpg')
h, w, _ = img.shape
new_img = np.zeros((h * 3, w * 3, 3), np.uint8)

for i in range(h * 3):
    for j in range(w * 3):
        new_img[i, j] = img[int(np.floor(i / 3)), int(np.floor(j / 3))]

cv2.imshow('2', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output/2.jpg', new_img)
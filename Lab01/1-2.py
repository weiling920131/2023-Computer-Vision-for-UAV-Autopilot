import cv2
import numpy as np

img = cv2.imread('images/nctu_flag.jpg')
img = np.array(img, dtype=np.int32)
new_img = img.copy()
h, w, _ = img.shape

contrast = 100
brightness = 40

for i in range(h):
    for j in range(w):
        B, G, R = img[i, j]
        if (B + G) * 0.3 > R:
            new_img[i, j] = (img[i, j] - 127) * (contrast / 127 + 1) + 127 + brightness

new_img = np.clip(new_img, 0, 255)
new_img = np.array(new_img, dtype=np.uint8)

cv2.imshow('1-2', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output/1-2.jpg', new_img)
import cv2
import numpy as np

img = cv2.imread('images/otsu.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape

count = [0] * 256
for i in range(h):
    for j in range(w):
        count[img[i, j]] += 1

hist = []
for i in range(256):
    hist += [i] * count[i]

minV = float('inf')
t = 0
for threshold in range(256-1):
    t += count[threshold]
    v = np.var(hist[:t]) + np.var(hist[t:])
    if v <= minV:
        minV = v
        minT = threshold

new_img = img.copy()
for i in range(h):
    for j in range(w):
        new_img[i, j] = 0 if new_img[i, j] <= minT else 255

cv2.imshow('3', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output/3.jpg', new_img)
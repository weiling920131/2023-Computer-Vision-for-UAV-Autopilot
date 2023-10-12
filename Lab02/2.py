import cv2
import numpy as np

def histogram(img, c):
    h, w, _ = img.shape
    hist = [0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j ,c]] += 1

    trans = [0] * 256
    for k in range(256):
        trans[k] = round(255 * sum(hist[0:k+1]) / (h * w))

    return np.array(trans)


img = cv2.imread('images/histogram.jpg')
h, w, _ = img.shape

# 2-a: BGR
new_img_a = img.copy()
trans_B = histogram(img, 0)
trans_G = histogram(img, 1)
trans_R = histogram(img, 2)

for i in range(h):
    for j in range(w):
        new_img_a[i, j] = [trans_B[img[i, j, 0]], trans_G[img[i, j, 1]], trans_R[img[i, j, 2]]]

cv2.imshow('2-a', new_img_a)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output/2-a.jpg', new_img_a)

# 2-b: HSV
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
new_img_b = img.copy()
trans_V = histogram(img, 2)

for i in range(h):
    for j in range(w):
        new_img_b[i, j, 2] = trans_V[img[i, j, 2]]

new_img_b = cv2.cvtColor(new_img_b, cv2.COLOR_HSV2BGR)

cv2.imshow('2-b', new_img_b)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output/2-b.jpg', new_img_b)
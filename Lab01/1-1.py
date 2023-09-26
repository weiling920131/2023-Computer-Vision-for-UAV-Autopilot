import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv2.imread('images/nctu_flag.jpg')
    # image shape return height, width, channel respectively
    # channel order is B, G, R
    (H, W, C) = img.shape
    for i in range(H):
        for j in range(W):
            if img[i, j][0] > 60 and img[i, j][0] * 0.8 > img[i, j][1] and img[i, j][0] * 0.8 > img[i, j][2]:
                continue
            else:
                gray = sum(img[i, j]) / 3
                img[i, j] = [gray, gray, gray]
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
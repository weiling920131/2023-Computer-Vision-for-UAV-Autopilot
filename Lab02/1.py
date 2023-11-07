import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def edge_detect(img):
    img = cv2.imread(img)
    cv2.imshow('Origin img', img)
    H_img, W_img, C_img = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_img', gray_img)
    denoise_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    cv2.imshow('denoise_img', denoise_img)
    x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_kernel = x_kernel.T
    # second papaer
    gx = cv2.filter2D(denoise_img, -1, x_kernel)
    gy = cv2.filter2D(denoise_img, -1, y_kernel)
    # make img array has larger buffer, avoid result of add overflow
    new_img = np.array((abs(gx) + abs(gy)), dtype=np.int32)
    np.clip(new_img, 0, 255, out=new_img)
    # finally, turn img into uint8, to fit the format(0-255)
    new_img = np.array(new_img, dtype=np.uint8)
    cv2.imwrite('result/edge.png', new_img)
    cv2.imshow('edge', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = './Frieren.jpg'
    edge_detect(img)
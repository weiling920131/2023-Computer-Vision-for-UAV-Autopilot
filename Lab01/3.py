import cv2
import os
import numpy as np
import math

def Biinterpolation(enlarge_size):
    result_path = './results'
    if not os.path.exists(result_path):
        # makedirs create directory recurisively, mkdir create directory
        os.makedirs(result_path)
    
    img = cv2.imread('images/nctu_flag.jpg')
    cv2.imshow('Origin img', img)
    H_img, W_img, C_img = img.shape
    # Zero padding
    # img shape: (H, W, C)
    img = np.vstack([img, np.zeros([1, W_img, C_img])])
    img = np.hstack([img, np.zeros([H_img+1, 1, C_img])])
    img_new = np.zeros([H_img*enlarge_size, W_img*enlarge_size, C_img], dtype=np.uint8) 
    H_new, W_new, C_new = img_new.shape
    for i in range(H_new): 
        for j in range(W_new):
            x1 = math.floor(i/enlarge_size)
            x2 = x1 + 1
            y1 = math.floor(j / enlarge_size)
            y2 = y1 + 1
            left = img[x1, y1]*(y2 - j / enlarge_size) + img[x1, y2]*(j/enlarge_size - y1)
            right = img[x2, y1]*(y2 - j / enlarge_size) + img[x2, y2]*(j/enlarge_size - y1)
            img_new[i, j] = left*(x2 - i/enlarge_size) + right*(i/enlarge_size - x1)
    cv2.imwrite('results/3.png', img_new)

if __name__ == '__main__':
    Biinterpolation(enlarge_size=3)
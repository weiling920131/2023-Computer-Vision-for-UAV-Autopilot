import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    result_path = './results'
    if not os.path.exists(result_path):
        # makedirs create directory recurisively, mkdir create directory
        os.makedirs(result_path)
    enlarge_size = 3
    img = cv2.imread('images/nctu_flag.jpg')
    cv2.imshow('Origin img', img)
    H_img, W_img, C_img = img.shape
    enlarge_img = np.zeros([H_img*enlarge_size, W_img*enlarge_size, 3], dtype=np.uint8)
    H_enlarege, W_enlarge, C_enlarge = enlarge_img.shape
    for i in range(H_enlarege): 
        for j in range(W_enlarge):
            y = int(i / enlarge_size)
            x = int(j / enlarge_size)
            enlarge_img[i, j] = img[y, x]
    cv2.imshow('resize img', enlarge_img)
    enlarge_path = os.path.join(result_path, 'enlarge_image.png')
    cv2.imwrite(enlarge_path, enlarge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
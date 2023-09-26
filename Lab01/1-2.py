import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    result_path = './results'
    if not os.path.exists(result_path):
        # makedirs create directory recurisively, mkdir create directory
        os.makedirs(result_path)

    # img read in is already a np array
    img = cv2.imread('images/nctu_flag.jpg')
    cv2.imshow('Origin img', img)
    # image shape return height, width, channel respectively
    # channel order is B, G, R
    (H, W, C) = img.shape
    # to prevent overflow, make img dtype can handle positive and negative value
    # and give 32 bit range to it
    contrast = 100
    brightness = 40
    img_array = np.array(img, dtype = np.int32)
    for i in range(H):
        for j in range(W):
            if (img_array[i, j][0] + img_array[i, j][1]) * 0.3 > img_array[i, j][2]:
                old_pixel = img_array[i, j]
                img_array[i, j] = (old_pixel - 127) * (contrast/127+1) + 127 + brightness
    img_array = np.clip(img_array, 0, 255)
    # give 8 bit, control range in (0, 255), ensure positive
    img = np.array(img_array, dtype=np.uint8)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
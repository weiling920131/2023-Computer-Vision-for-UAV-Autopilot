import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def make_histogram(img, mode=256):
    h, w = img.shape
    histogram = np.zeros(mode)
    for i in range(h):
        for j in range(w):
            histogram[img[i][j]] += 1
            
    return histogram

def find_min_threshold(histogram):
    Min = sys.maxsize
    min_threshold = 255
    for threshold in range(1, 255):
        lower = np.array([])
        larger = np.array([])

        for i in range(threshold+1):
            lower = np.append(lower, np.full(int(histogram[i]), i))
        for i in range(threshold+1, 256):
            larger = np.append(larger, np.full(int(histogram[i]), i))

        if len(lower) == 0 or len(larger) == 0:
            continue
        
        var_low = np.var(np.array(lower))
        var_lar = np.var(np.array(larger))
        if Min >= var_low + var_lar:
            Min = var_lar + var_low
            min_threshold = threshold

    return min_threshold

if __name__ == '__main__':
    img = cv2.imread('images/otsu.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    
    histogram = make_histogram(img)
    min_threshold = find_min_threshold(histogram)

    print(min_threshold)
    for i in range(h):
        for j in range(w):
            img[i][j] = 0 if img[i][j] <= min_threshold else 255
    
    cv2.imwrite('output/3.png', img)
    cv2.imshow('img', img)
    cv2.waitKey(0)
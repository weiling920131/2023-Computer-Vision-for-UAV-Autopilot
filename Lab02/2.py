import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def transform_channel(img, channel):
    H, W, _ = img.shape
    total_pixels = 256
    histogram = np.zeros(total_pixels)
    transform_result = np.zeros(total_pixels)
    for h in range(H):
        for w in range(W):
            # perform in index 
            histogram[img[h, w, channel]] += 1
    for i in range(total_pixels):
        transform_result[i] = (total_pixels-1) * (sum(histogram[:i+1]) / (H * W))
    np.clip(transform_result, 0, 255, out=transform_result)
    transform_result = np.array(transform_result, dtype=np.uint8)
    return transform_result
    
def get_histoEQ_result(img):
    H, W, C = img.shape
    new_img = np.zeros([H, W, C], dtype=np.uint8)
    for c in range(C):
        transform_result = transform_channel(img, c)
        # assign new value to img
        for h in range(H):
            for w in range(W):
                # perform in index 
                new_img[h, w, c] = transform_result[img[h, w, c]]
    return new_img

def histogram_equal(img):
    img = cv2.imread(img)
    new_img = get_histoEQ_result(img)

    cv2.imwrite('result/2-a.png', new_img)
    cv2.imshow('histogram-equal', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def hsv_histo_eqal(img):
    img = cv2.imread(img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W, C = img.shape
    transform_result = transform_channel(img, 2)
    # assign new value to img
    for h in range(H):
        for w in range(W):
            # perform in index 
            hsv_img[h, w, 2] = transform_result[img[h, w, 2]]
    new_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    cv2.imwrite('result/2-b.png', new_img)
    cv2.imshow('HSV-histogram-equal', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    img = 'histogram.jpg'
    
    histogram_equal(img)
    hsv_histo_eqal(img)
    
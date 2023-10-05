import cv2
import numpy as np

def make_transformation(img, color, mode=256):
    h, w, _ = img.shape
    histogram = np.zeros(mode)
    for i in range(h):
        for j in range(w):
            histogram[img[i][j][color]] += 1
    trans = np.zeros(mode)
    for i in range(mode):
        trans[i] = (mode-1)  * sum(histogram[0:i+1])/(h*w)

    return trans

def BGR_histogram(img):
    h, w, _ = img.shape
    new_img = np.zeros([h, w, 3], dtype=np.uint8)
    b_trans = make_transformation(img, 0)
    g_trans = make_transformation(img, 1)
    r_trans = make_transformation(img, 2)

    for i in range(h):
        for j in range(w):
            new_img[i][j] = [b_trans[img[i][j][0]], g_trans[img[i][j][1]], r_trans[img[i][j][2]]]

    return new_img

def HSV_histogram(img):
    h, w, _ = img.shape
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    v_trans = make_transformation(new_img, 2)
    for i in range(h):
        for j in range(w):
            new_img[i][j][2] = v_trans[new_img[i][j][2]]

    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
    return new_img
    

if __name__ == '__main__':
    img = cv2.imread('images/histogram.jpg')
    bgr_img = BGR_histogram(img)
    cv2.imwrite('output/2-a.png', bgr_img)
    cv2.imshow('img', bgr_img)
    cv2.waitKey(0)

    hsv_img = HSV_histogram(img)
    cv2.imwrite('output/2-b.png', hsv_img)
    cv2.imshow('img', hsv_img)
    cv2.waitKey(0)
import cv2
import numpy as np
import math

def bilinear(image, rate):
    img = cv2.imread(image)
    
    x, y, _ = img.shape
    img = np.vstack([img,np.zeros([1, y, 3])])
    img = np.hstack([img,np.zeros([x + 1, 1, 3])])
    new_img = np.zeros([x*rate, y*rate, 3], dtype=int)
    for i in range(x*rate):
        for j in range(y*rate):
            x1 = math.floor(i/rate)
            x2 = x1 + 1
            y1 = math.floor(j/rate)
            y2 = y1 + 1

            top = img[x1, y1]*(y2 - j/rate) + img[x1, y2]*(j/rate - y1)
            bottom = img[x2, y1]*(y2 - j/rate) + img[x2, y2]*(j/rate - y1)

            new_img[i, j] = top*(x2 - i/rate) + bottom*(i/rate - x1) 

    new_img = new_img.astype(np.uint8)
    cv2.imwrite('./output/3.png', new_img)
    cv2.imshow('img',new_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    image = './images/test.jpg'
    rate = 3
    bilinear(image, rate)

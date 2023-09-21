import cv2
import numpy as np

def Bilinear(img, rate):
    h, w, _ = img.shape
    new_img = np.zeros((h * rate, w * rate, 3), np.uint8)

    for i in range(h * rate):
        for j in range(w * rate):
            y = i / rate
            x = j / rate
            y1 = int(np.floor(y)) if int(np.floor(y)) < h else h - 1
            y2 = y1 + 1 if y1 + 1 < h else h - 1
            x1 = int(np.floor(x)) if int(np.floor(x)) < w else w - 1
            x2 = x1 + 1 if x1 + 1 < w else w - 1

            A = img[y1, x1] * (x2 - x) + img[y1, x2] * (x - x1)
            B = img[y2, x1] * (x2 - x) + img[y2, x2] * (x - x1)
            new_img[i, j] = A * (y2 - y) + B * (y - y1)

    cv2.imshow('3', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('output/3.jpg', new_img)

if __name__ == "__main__":
    img = cv2.imread('images/test.jpg')
    rate = 3
    Bilinear(img, rate)
import cv2
import numpy as np

img = cv2.imread('images/screen.jpg')
img_corner = np.float32([[276, 190], [618, 86], [263, 402], [627, 370]])

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    h, w, _ = frame.shape
    cap_corner = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    frame_padding = np.zeros((h + 1, w + 1, 3), np.uint8)
    frame_padding[0:h, 0:w] = frame

    proj = cv2.getPerspectiveTransform(img_corner, cap_corner)
    new_img = img.copy()

    for y in range(86, 402+1):
        for x in range(263, 627+1):
            loc = np.matmul(proj, [x, y, 1])
            loc /= loc[2]
            loc = [int(loc[0]), int(loc[1])]  # cap [x, y]
            
            if 0 <= loc[0] < w and 0 <= loc[1] < h:
                x1 = loc[0]
                x2 = x1 + 1
                y1 = loc[1]
                y2 = y1 + 1

                A = frame_padding[y1, x1] * (x2 - loc[0]) + frame_padding[y1, x2] * (loc[0] - x1)
                B = frame_padding[y2, x1] * (x2 - loc[0]) + frame_padding[y2, x2] * (loc[0] - x1)
                new_img[y, x] = A * (y2 - loc[1]) + B * (loc[1] - y1)
                
    cv2.imshow('new img', new_img)
    key = cv2.waitKey(33)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
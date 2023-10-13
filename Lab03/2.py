import cv2
import numpy as np

cap = cv2.VideoCapture('images/train.mp4')
backSub = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break

    fgmask = backSub.apply(frame)

    shadowval = backSub.getShadowValue()
    ret, nmask = cv2.threshold(fgmask, shadowval, 255, cv2.THRESH_BINARY)

    h, w = nmask.shape
    label = np.zeros((h, w))
    equal = np.array([[None] * (h * w)]) 
    # break
    for i in range(h):
        for j in range(w):
            if label[i, j] == 0 and nmask[i, j] == 255:
                if i == 0 and j == 0:
                    label[i, j] = 1
                elif i == 0:
                    if label[i, j-1] != 0:
                        label[i, j] = label[i, j-1]
                    else:
                        label[i, j] = 1
                elif j == 0:
                    if label[i-1, j] != 0:
                        label[i, j] = label[i-1, j]
                    else:
                        label[i, j] = 1
                else:
                    if label[i, j-1] != 0 and label[i-1, j] != 0:
                        label[i, j] = label[i, j-1]
                        if label[i, j-1] != label[i-1, j]:
                            equal[label[i, j-1]] += label[i-1, j]
                            equal[label[i-1, j]] += label[i, j-1]


    cv2.imshow("frame", frame)
    cv2.imshow("mask", nmask)
    cv2.waitKey(33)

cap.release()
cv2.destroyAllWindows()
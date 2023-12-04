import cv2
import numpy as np

# Haar-cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar-cascade
    rects = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in rects:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # HOG
    rects, weights = hog.detectMultiScale(frame_gray, winStride=(8, 8), scale=1.1, useMeanshiftGrouping=False)

    for (x, y, w, h) in rects:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('img', img)
    key = cv2.waitKey(33)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

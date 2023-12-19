import cv2
import numpy as np


cap = cv2.VideoCapture(0)
fs = cv2.FileStorage("test.xml", cv2.FILE_STORAGE_READ)
cameraMatrix = fs.getNode("intrinsic").mat()
distCoeffs = fs.getNode("distortion").mat()

face_objectPoints = np.array([[0,0,0],[13,0,0],[0,17,0],[13,17,0]], dtype=np.float32)
ped_objectPoints = np.array([[0,0,0],[0.5,0,0],[0,1.75,0],[0.4,1.61,0]], dtype=np.float32)

while(1):
    ret, frame = cap.read()
    if not ret: break
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rects = face_cascade.detectMultiScale(frame, 1.15, 3, minSize=(100,100))
    for x,y,w,h in rects:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        retval, rvec, tvec = cv2.solvePnP(face_objectPoints, np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]], dtype=np.float32), cameraMatrix, distCoeffs)
        if(not retval):
            continue
        x_text = "x: " + str(tvec[1])
        z_text = " z: " + str(tvec[2]) 
        cv2.putText(frame, x_text, (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(frame, z_text, (x, y+h+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.waitKey(33)
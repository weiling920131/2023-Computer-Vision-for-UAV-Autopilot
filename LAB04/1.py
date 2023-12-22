import cv2
import numpy as np

cap = cv2.VideoCapture(0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objectPoints = []
imagePoints = []
patternSize = (9*20, 6*20)
winSize = (11, 11)
zeroZone = (-1, -1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
cnt = 0
while(1):
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imageSize = frame.shape[::-1]
    ret, corner = cv2.findChessboardCorners(frame, (9,6), None)
    cv2.imshow('frame', frame)
    cv2.waitKey(33)
    if not ret: continue
    objectPoints.append(objp)
    corners = cv2.cornerSubPix(frame, corner, winSize, zeroZone, criteria)
    imagePoints.append(corners)
    cnt+=1
    print(cnt)
    if cnt == 20: break
print('work\n')
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize, None, None)

f = cv2.FileStorage('test.xml', cv2.FILE_STORAGE_WRITE)
f.write("intrinsic", cameraMatrix) 
f.write("distortion", distCoeffs)
f.release()
import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

def main():
    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()
    
    patternSize = (9, 6)
    objectPoints = []
    for i in range(9):
        for j in range(6):
            objectPoints += [[i, j, 0]]
    objectPoints = np.array([objectPoints] * 10, dtype=np.float32)
    print(objectPoints.shape)

    imagePoints = []

    while True:
        frame = frame_read.frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = frame.shape
        imageSize = (w, h)

        ret, corner = cv2.findChessboardCorners(frame, patternSize, None)
        if ret:
            corner = cv2.cornerSubPix(frame, corner, (11, 11), (-1, -1),  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            corner = corner.T.reshape(-1, 2)
            imagePoints.append(corner)
            # print(corner.shape)
            if len(imagePoints) >= 10:
                break
        
        cv2.imshow("drone", frame)
        key = cv2.waitKey(500)
    
    imagePoints = np.array(imagePoints)
    print(imagePoints.shape)

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize, None, None)
    # print(cameraMatrix, distCoeffs, rvecs, tvecs)
    f = cv2.FileStorage('calibrateCamera.xml', cv2.FILE_STORAGE_WRITE)
    f.write("intrinsic", cameraMatrix)
    f.write("distortion", distCoeffs)
    f.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

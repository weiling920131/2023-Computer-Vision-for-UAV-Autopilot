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
    
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    fs = cv2.FileStorage("calibrateCamera.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()

    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        if markerIds is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
            print(rvec)
            print(tvec, '\n')

            # for i in range(len(markerIds)):
            frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec[0], tvec[0], 0.1)
            text = "x:" + str(tvec[0, 0, 0]) + " y:" + str(tvec[0, 0, 1]) + " z: " + str(tvec[0, 0, 2])
            
            # R, _ = cv2.Rodrigues(rvec[0])
            # V = np.matmul(R, [0, 0, 1])
            # rad = math.atan(V[0]/V[2])
            # deg = rad / math.pi * 180
            # print("deg: ", deg)

            cv2.putText(frame, text, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("drone", frame)
        key = cv2.waitKey(33)
    
    #cv2.destroyAllWindows()



if __name__ == '__main__':
    main()


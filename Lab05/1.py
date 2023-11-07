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
    
    fs = cv2.FileStorage("output.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()

    while True:
        frame = frame_read.frame
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters_create()

        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        
        rvec, tvec, _objPoints =cv2.aruco.estimatePoseSingleMarkers(markerCorners,15, intrinsic, distortion)
        frame = cv2.aruco.drawAxis(frame, intrinsic,distortion, rvec, tvec, 0.1)
        cv2.imshow("drone", frame)
        key = cv2.waitKey(33)
    
    #cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

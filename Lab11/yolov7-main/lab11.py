from numpy import random
import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import  plot_one_box

import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

horizontal = [
    [[1, 1, 1],
     [0, 0, 0],
     [0, 0, 0]],

    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]],

    [[0, 0, 0],
     [0, 0, 0],
     [1, 1, 1]]
]

vertical = [
    [[1, 0, 0],
     [1, 0, 0],
     [1, 0, 0]],

    [[0, 1, 0],
     [0, 1, 0],
     [0, 1, 0]],

    [[0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]]
]

corner_ul = [
    [[0, 1, 0],
     [1, 1, 0], 
     [0, 0, 0]],

    [[1, 0, 0],
     [0, 0, 0], 
     [0, 0, 0]],

    [[1, 1, 0], 
     [0, 0, 0], 
     [0, 0, 0]],

    [[1, 0, 0], 
     [1, 0, 0], 
     [0, 0, 0]],

    [[0, 0, 1],
     [1, 1, 1],
     [0, 0, 0]],

    [[0, 1, 0], 
     [0, 1, 0], 
     [1, 1, 0]],

    [[0, 0, 1], 
     [0, 0, 1], 
     [1, 1, 1]]
]

corner_ur = [
    [[0, 1, 0],
     [0, 1, 1], 
     [0, 0, 0]],

    [[1, 0, 0],
     [1, 0, 0], 
     [1, 1, 1]],

    [[0, 1, 0],
     [0, 1, 0], 
     [0, 1, 1]],

    [[1, 0, 0],
     [1, 1, 1], 
     [0, 0, 0]],

    [[0, 0, 1],
     [0, 0, 1], 
     [0, 0, 0]],

    [[0, 1, 1],
     [0, 0, 0], 
     [0, 0, 0]],

    [[0, 0, 1],
     [0, 0, 0], 
     [0, 0, 0]]
]

corner_dl = [
    [[0, 0, 0],
     [1, 1, 0],
     [0, 1, 0]],

    [[1, 1, 0],
     [0, 1, 0], 
     [0, 1, 0]],

    [[1, 1, 1],
     [0, 0, 1], 
     [0, 0, 1]],

    [[0, 0, 0],
     [1, 0, 0], 
     [1, 0, 0]],

    [[0, 0, 0],
     [1, 1, 1], 
     [0, 0, 1]],

    [[0, 0, 0],
     [0, 0, 0], 
     [1, 0, 0]],

    [[0, 0, 0],
     [0, 0, 0], 
     [1, 1, 0]]
]

corner_dr = [
    [[0, 0, 0],
     [0, 1, 1], 
     [0, 1, 0]],

    [[1, 1, 1],
     [1, 0, 0], 
     [1, 0, 0]],

    [[0, 1, 1],
     [0, 1, 0], 
     [0, 1, 0]],

    [[0, 0, 0],
     [1, 1, 1], 
     [1, 0, 0]],

    [[0, 0, 0],
     [0, 0, 1], 
     [0, 0, 1]],

    [[0, 0, 0],
     [0, 0, 0], 
     [0, 1, 1]],

    [[0, 0, 0],
     [0, 0, 0], 
     [0, 0, 1]]
]

corner_ulr = [
    [[0, 1, 0],
     [1, 1, 1], 
     [0, 0, 0]],

    [[1, 0, 0],
     [1, 0, 0], 
     [1, 1, 1]],

    [[0, 1, 0],
     [0, 1, 0], 
     [1, 1, 1]],

    [[0, 0, 1],
     [0, 0, 1], 
     [1, 1, 1]],

    [[1, 0, 0],
     [1, 1, 1], 
     [0, 0, 0]],

    [[0, 0, 1],
     [1, 1, 1], 
     [0, 0, 0]]
]


def main():
    # Tello
    drone = Tello()
    drone.connect()
    time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()
    
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    fs = cv2.FileStorage("test.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()
    fs.release()

    z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

    z_pid.initialize()
    y_pid.initialize()
    x_pid.initialize()
    yaw_pid.initialize()

    flag = 1
    path = 0
    # cap = cv2.VideoCapture(0)

    face_objectPoints = np.array([[0,0,0],[13,0,0],[0,17,0],[13,17,0]], dtype=np.float32)
    ped_objectPoints = np.array([[0,0,0],[0.5,0,0],[0,1.75,0],[0.4,1.61,0]], dtype=np.float32)
    while True:
        # print("check")
        key = cv2.waitKey(1)
        if key != -1:
            if key == ord('q'):
                break
            #keyboard(drone, key)
        else:
            frame = frame_read.frame
            # ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ####################
            # aruco detect
            ####################
            markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
            if markerIds is not None:
                frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
                rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)

                for i in range(len(markerIds)):
                    id = markerIds[i][0]
                    # TODO: detect different different id
                    if id == 1 and flag == 1:
                        z_update = tvec[i, 0, 2] - 60 # land region is 150 cm away from aruco-3
                        y_update = -(tvec[i, 0, 1] + 20)
                        x_update = tvec[i, 0, 0]
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = y_pid.update(y_update, sleep=0)
                        x_update = x_pid.update(x_update, sleep=0)

                        R, _ = cv2.Rodrigues(rvec[i])
                        V = np.matmul(R, [0, 0, 1])
                        rad = math.atan(V[0]/V[2])
                        deg = rad / math.pi * 180
                        yaw_update = yaw_pid.update(deg, sleep=0)
                        # if match rule, then land! 
                        # congrauation, you finish final!
                        if abs(x_update) <= 10 and abs(z_update) <= 10 and abs(yaw_update) <= 5:
                            # TODO: put detect result to doll
                            doll = "Melody"
                            flag = 2    
                            if doll == "Melody":
                                path = 1
                            elif doll == "Cana":
                                path = 2
                            break

                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = int(mss(x_update))
                            yaw_update = int(mss(yaw_update))
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                        
                    
                    if id == 2 and flag == 2:

                    if id == 3 and flag == 4:
                        z_update = tvec[i, 0, 2] - 150 # land region is 150 cm away from aruco-3
                        y_update = -(tvec[i, 0, 1] + 20)
                        x_update = tvec[i, 0, 0]
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = y_pid.update(y_update, sleep=0)
                        x_update = x_pid.update(x_update, sleep=0)

                        R, _ = cv2.Rodrigues(rvec[i])
                        V = np.matmul(R, [0, 0, 1])
                        rad = math.atan(V[0]/V[2])
                        deg = rad / math.pi * 180
                        yaw_update = yaw_pid.update(deg, sleep=0)
                        # if match rule, then land! 
                        # congrauation, you finish final!
                        if abs(x_update) <= 10 and abs(z_update) <= 10 and abs(yaw_update) <= 5:
                            drone.land()
                            break

                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = int(mss(x_update))
                            yaw_update = int(mss(yaw_update))
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                    else:
                        if flag == 4: # if not detect aruco-id-3, then turn left
                            drone.send_rc_control(0, 0, 0, -10)
            else: # if does't detect aruco, check with flag
                if flag == 2:
                    if path == 1: # Melody
                        # TODO: Second don't go up
                        pass
                    elif path == 2: # Cana
                        # TODO: first don't go up
                if flag == 3:
                    ####################
                    # face detect
                    ####################
                    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    rects = face_cascade.detectMultiScale(frame, 1.15, 3, minSize=(100,100))
                    face2_num = 0
                    faces_x = []
                    for x,y,w,h in rects:
                        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                        retval, rvec, tvec = cv2.solvePnP(face_objectPoints, np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]], dtype=np.float32), cameraMatrix, distCoeffs)
                        if(not retval):
                            continue
                        # if has detect face:
                        # align z: 60, x: 15
                        faces_x.append(tvec)
                        face2_num += 1
                        x_text = "x: " + str(tvec[0])
                        z_text = " z: " + str(tvec[2]) 
                        cv2.putText(frame, x_text, (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, z_text, (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        
                    if face2_num == 2: # if found 2 faces, then use 2 x_update to 
                        # frame.shape[0]: height, frame.shape[1]: weight
                        middle = int((faces_x[0][0]+ faces_x[1][0]) / 2)
                        x_update = abs(middle+15)
                        x_update = x_pid.update(x_update, sleep=0)
                        line_x = abs(middle+15) + int(frame.shape[1] / 2)
                        print(f"Detect 2 faces! middle: {middle}")
                        # cv2.line(frame, (middle, 0), (middle, frame.shape[0]), (255, 0, 0), 2)  # (255, 0, 0) is the color (BGR format), 2 is the line thickness
                        # line start(x, y), end(x,y)
                        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)  # (255, 0, 0) is the color (BGR format), 2 is the line thickness
                        if x_update < 5:
                            #TODO go to flag 4
                            flag = 4
                            # go forward for a while
                            # drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                            drone.send_rc_control(0, 50, 0, 0)
                            time.sleep(2)
                        else:
                            z_update = 0
                            y_update = 0
                            x_update = int(mss(x_update))
                            yaw_update = 0
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                    else:
                        # if not found 2 faces yet, keep go to right
                        drone.send_rc_control(10, 0, 0, 0)
                                        
            cv2.imshow('frame', frame)
            cv2.waitKey(33)
            
if __name__ == '__main__':
    main()
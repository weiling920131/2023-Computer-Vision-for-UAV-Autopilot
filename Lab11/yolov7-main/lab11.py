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

import lab11info
horizontal = lab11info.get_horizontal()
vertical = lab11info.get_vertical()
corner_ul = lab11info.get_corner_ul()
corner_ur = lab11info.get_corner_ur()
corner_dl = lab11info.get_corner_dl()
corner_dr = lab11info.get_corner_dr()
corner_ulr = lab11info.get_corner_ulr()
corner_cana1 = lab11info.get_corner_cana1()
corner_cana2 = lab11info.get_corner_cana2()



WEIGHT = './runs/train/yolov7-lab09/weights/best.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"

model = attempt_load(WEIGHT, map_location=device)
if device == "cuda":
    model = model.half().to(device)
else:
    model = model.float().to(device)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

def mss(update, max_speed_threshold=40):
    if update > max_speed_threshold:
        update = max_speed_threshold
    elif update < -max_speed_threshold:
        update = -max_speed_threshold

    return update

def main():
    # Tello
    drone = Tello()
    drone.connect()
    time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()
    
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    fs = cv2.FileStorage("calibrateCamera.xml", cv2.FILE_STORAGE_READ)
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

    flag = 0
    face_flag = 0
    path = 0
    path_flag = 0
    doll = ''
    handle = False

    face_objectPoints = np.array([[0,0,0],[13,0,0],[0,17,0],[13,17,0]], dtype=np.float32)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # ped_objectPoints = np.array([[0,0,0],[0.5,0,0],[0,1.75,0],[0.4,1.61,0]], dtype=np.float32)
    while True:
        
        key = cv2.waitKey(1)
        if key != -1:
            if key == ord('m'):
                handle = not handle
            elif key == ord('q'):
                break
            keyboard(drone, key)
        else:
            if handle:
                print("handle++++++++++++++++++++++")
                drone.send_rc_control(0, 0, 0, 0)
                continue
            print("\n\npath_flag)))))))))))))))))))))))))))))))))))))))))))))) ",path_flag)
            print("\n\nflag===========================", flag,'\n\n')
            print("doll++++++++++++++",doll)
            print("path:",path)
            
            frame = frame_read.frame
            if flag == 0: # determine if it is melody or cana
                frame1 = frame
                image_orig = frame1.copy()
                frame1 = letterbox(frame1, (640, 640), stride=64, auto=True)[0]
                if device == "cuda":
                    frame1 = transforms.ToTensor()(frame1).to(device).half().unsqueeze(0)
                else:
                    frame1 = transforms.ToTensor()(frame1).to(device).float().unsqueeze(0)

                with torch.no_grad():
                    output = model(frame1)[0]
                output = non_max_suppression_kpt(output, conf_thres=0.25, iou_thres=0.65)[0]
                ## Return: list of detections, on (n,6) tensor per image [xyxy, confidance, class]
                
                ## Draw label and confidence on the image
                output[:, :4] = scale_coords(frame1.shape[2:], output[:, :4], image_orig.shape).round()
                for *xyxy, conf, cls in output:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    doll = label.split()[0]
                    if float(label.split()[1]) > 0.7:
                        flag = 1
                        break
            
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
                    if id == 1 and (flag == 1 or flag == 0):
                            
                        z_update = tvec[i, 0, 2] - 70
                        y_update = -(tvec[i, 0, 1] + 20)
                        x_update = tvec[i, 0, 0]
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = y_pid.update(y_update, sleep=0)
                        x_update = x_pid.update(x_update, sleep=0)

                        # R, _ = cv2.Rodrigues(rvec[i])
                        # V = np.matmul(R, [0, 0, 1])
                        # rad = math.atan(V[0]/V[2])
                        # deg = rad / math.pi * 180
                        # yaw_update = yaw_pid.update(deg, sleep=0)

                        if abs(x_update) <= 10 and abs(z_update) <= 10 and flag == 1:

                            flag = 2    
                            if doll == "melody":
                                path = 1
                            elif doll == "cana":
                                path = 2

                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = int(mss(x_update))
                            yaw_update = 0#int(mss(yaw_update)+3)
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                        break
                    
                    if id == 2 and (flag == 2 or flag == 3):
                        if flag == 2:
                            flag = 3                        
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(0.5)

                        z_update = tvec[i, 0, 2] - 60 
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

                        if abs(x_update) <= 10 and abs(z_update) <= 10 and abs(yaw_update) <= 2:

                            drone.send_rc_control(0, 0,0, 0)
                            time.sleep(0.5)
                            drone.rotate_clockwise(85)
                            time.sleep(0.5)
                            drone.send_rc_control(0,0,0,0)
                            time.sleep(0.5)
                            drone.send_rc_control(0,-25,0,0)
                            time.sleep(2)
                            drone.send_rc_control(30,0,0,0)
                            time.sleep(3)
                            drone.send_rc_control(0,0,0,0)
                            time.sleep(0.5)
                            # drone.move_back(100)
                            flag = 33

                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = int(mss(x_update))
                            yaw_update = int(mss(yaw_update)-1)
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                        break

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
                        if abs(x_update) <= 10 and abs(z_update) <= 10 and abs(yaw_update) <= 2:
                            drone.land()
                            break

                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = int(mss(x_update))
                            yaw_update = int(mss(yaw_update))
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                    else:
                        if flag == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            
                            gray = cv2.bitwise_not(gray)
                            gray = cv2.erode(gray, (7, 7), iterations=3)
                            gray = cv2.dilate(gray, (7, 7), iterations=3)
                            gray = cv2.bitwise_not(gray)
                            h, w = gray.shape

                            gray = gray[int(h/5):int(h/5*4), int(w/5):int(w/5*4)]
                            h, w = gray.shape
                            gray = cv2.GaussianBlur(gray, (7, 7), 0)

                            _, frame = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

                            if path == 1 and path_flag == 7:
                                black = lab11info.line_follower(frame ,h, w, 0.3)
                            elif path == 2 and path_flag == 3:
                                black = lab11info.line_follower(frame ,h, w, 0.3)
                            black = lab11info.line_follower(frame, h, w)

                            speed = 10
                            y_speed = 25
                            # print(black, '\n')
                            if black == [[1, 1, 1],
                                        [1, 1, 1], 
                                        [1, 1, 1]]:
                                drone.send_rc_control(0, -speed, 0, 0)  # too close, go back
                                
                            # Melody
                            if path == 1:
                                if path_flag == 0 and (black in corner_dl or black in horizontal):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 1 and (black in corner_ulr or black in vertical):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(0, 0, y_speed, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 2 and (black in corner_dl or black in horizontal):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 3 and (black in corner_dr or black in vertical):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(0, 0, -y_speed, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 4 and (black in corner_ulr or black in horizontal or black in corner_cana1):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 5 and (black in corner_dr or black in vertical or black in corner_cana1):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(0, 0, -y_speed, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 6 and (black in corner_ul or black in horizontal):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 7 and (black in corner_ur or black in vertical):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(0, 0, y_speed, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 8 and (black in corner_dl or black in horizontal):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                else:
                                    if path_flag == 0 or path_flag == 2 or path_flag == 8:  # up
                                        drone.send_rc_control(0, 0, y_speed, 0)
                                    elif path_flag == 1 or path_flag == 3 or path_flag == 5 or path_flag == 9:    # left
                                        drone.send_rc_control(-speed, 0, 0, 0)
                                    elif path_flag == 4 or path_flag == 6:    # down
                                        drone.send_rc_control(0, 0, -y_speed, 0)
                                    elif path_flag == 7:    #left (under table)
                                        drone.send_rc_control(-speed, 0, 0, 0)
                                        # time.sleep(2)
                                    else:
                                        if black == [[0, 0, 0],
                                                    [0, 0, 0], 
                                                    [0, 0, 0]]:
                                            # print("stop")
                                            drone.send_rc_control(0, 0, 0, 0)

                            # Cana
                            elif path == 2:
                                if path_flag == 0 and (black in corner_dl or black in horizontal):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 1 and (black in corner_dr or black in corner_cana1):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    # drone.send_rc_control(speed, 0, 0, 0)
                                    # time.sleep(0.5)
                                    drone.send_rc_control(0, 0, -y_speed, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 2 and (black in corner_ul or black in horizontal):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 3 and (black in corner_ur or black in vertical):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    drone.send_rc_control(0, 0, y_speed, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 4 and (black in corner_dl or black in horizontal):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    path_flag += 5

                                elif path_flag == 5 and (black in corner_ulr or black in vertical):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(0, 0, y_speed, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 6 and (black in corner_dl or black in horizontal):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 7 and (black in corner_dr or black in vertical):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(0, 0, -y_speed, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 8 and (black in corner_ulr or black in horizontal):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                else:
                                    if path_flag == 0 or path_flag == 4 or path_flag == 6:  # up
                                        drone.send_rc_control(0, 0, y_speed, 0)
                                    elif path_flag == 1 or path_flag == 5 or path_flag == 7 or path_flag == 9:    # left
                                        drone.send_rc_control(-speed, 0, 0, 0)
                                    elif path_flag == 2 or path_flag == 8:    # down
                                        drone.send_rc_control(0, 0, -y_speed, 0)
                                    elif path_flag == 3:    #left (under table)
                                        drone.send_rc_control(-speed, 0, 0, 0)
                                        # time.sleep(2)
                                    else:
                                        if black == [[0, 0, 0],
                                                    [0, 0, 0], 
                                                    [0, 0, 0]]:
                                            # print("stop")
                                            drone.send_rc_control(0, 0, 0, 0)
                        else:
                            drone.send_rc_control(0, 0, 0, 0)
            else: # if does't detect aruco, check with flag
                if flag == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    gray = cv2.bitwise_not(gray)
                    gray = cv2.erode(gray, (7, 7), iterations=3)
                    gray = cv2.dilate(gray, (7, 7), iterations=3)
                    gray = cv2.bitwise_not(gray)
                    h, w = gray.shape

                    gray = gray[int(h/5):int(h/5*4), int(w/5):int(w/5*4)]
                    h, w = gray.shape
                    gray = cv2.GaussianBlur(gray, (7, 7), 0)

                    _, frame = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

                    if path == 1 and path_flag == 7:
                        black = lab11info.line_follower(frame ,h, w, 0.3)
                    elif path == 2 and path_flag == 3:
                        black = lab11info.line_follower(frame ,h, w, 0.3)
                    black = lab11info.line_follower(frame, h, w)

                    speed = 10
                    y_speed = 25
                    # print(black, '\n')
                    if black == [[1, 1, 1],
                                [1, 1, 1], 
                                [1, 1, 1]]:
                        drone.send_rc_control(0, -speed, 0, 0)  # too close, go back
                        
                    # Melody
                    if path == 1:
                        if path_flag == 0 and (black in corner_dl or black in horizontal):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 1 and (black in corner_ulr or black in vertical):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(0, 0, y_speed, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 2 and (black in corner_dl or black in horizontal):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 3 and (black in corner_dr or black in vertical):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(0, 0, -y_speed, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 4 and (black in corner_ulr or black in horizontal or black in corner_cana1):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(0, 0, -5, 0)
                            time.sleep(0.2)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 5 and (black in corner_dr or black in vertical or black in corner_cana1):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(0, 0, -y_speed, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 6 and (black in corner_ul or black in horizontal):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 7 and (black in corner_ur or black in vertical):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(speed, 0, 0, 0)
                            time.sleep(0.5)
                            drone.send_rc_control(0, 0, y_speed, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 8 and (black in corner_dl or black in horizontal or black in corner_cana2):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        else:
                            if path_flag == 0 or path_flag == 2 or path_flag == 8:  # up
                                drone.send_rc_control(0, 0, y_speed, 0)
                            elif path_flag == 1 or path_flag == 3 or path_flag == 5 or path_flag == 9:    # left
                                drone.send_rc_control(-speed, 0, 0, 0)
                            elif path_flag == 4 or path_flag == 6:    # down
                                drone.send_rc_control(0, 0, -y_speed, 0)
                            elif path_flag == 7:    #left (under table)
                                drone.send_rc_control(-speed, 0, 0, 0)
                                # time.sleep(2)
                            else:
                                if black == [[0, 0, 0],
                                            [0, 0, 0], 
                                            [0, 0, 0]]:
                                    # print("stop")
                                    drone.send_rc_control(0, 0, 0, 0)

                    # Cana
                    elif path == 2:
                        if path_flag == 0 and (black in corner_dl or black in horizontal):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 1 and (black in corner_dr or black in corner_cana1):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            # drone.send_rc_control(speed, 0, 0, 0)
                            # time.sleep(0.5)
                            drone.send_rc_control(0, 0, -y_speed, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 2 and (black in corner_ul or black in horizontal):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 3 and (black in corner_ur or black in vertical):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            drone.send_rc_control(0, 0, y_speed, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 4 and (black in corner_dl or black in horizontal or black in corner_cana2):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            path_flag += 5

                        elif path_flag == 5 and (black in corner_ulr or black in vertical or black in corner_cana2):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(0, 0, y_speed, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 6 and (black in corner_dl or black in horizontal):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 7 and (black in corner_dr or black in vertical):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(0, 0, -y_speed, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 8 and (black in corner_ulr or black in horizontal):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        else:
                            if path_flag == 0 or path_flag == 4 or path_flag == 6:  # up
                                drone.send_rc_control(0, 0, y_speed, 0)
                            elif path_flag == 1 or path_flag == 5 or path_flag == 7 or path_flag == 9:    # left
                                drone.send_rc_control(-speed, 0, 0, 0)
                            elif path_flag == 2 or path_flag == 8:    # down
                                drone.send_rc_control(0, 0, -y_speed, 0)
                            elif path_flag == 3:    #left (under table)
                                drone.send_rc_control(-speed, 0, 0, 0)
                                # time.sleep(2)
                            else:
                                if black == [[0, 0, 0],
                                            [0, 0, 0], 
                                            [0, 0, 0]]:
                                    # print("stop")
                                    drone.send_rc_control(0, 0, 0, 0)
                elif flag == 33:
                    rects = face_cascade.detectMultiScale(frame, 1.15, 6, minSize=(50,50))
                    face2_num = 0
                    faces_x = []
                    for x,y,w,h in rects:
                        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                        retval, rvec, tvec = cv2.solvePnP(face_objectPoints, np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]], dtype=np.float32), intrinsic, distortion)
                        if not retval:
                            continue
                        print(tvec)

                        faces_x.append(tvec)
                        face2_num += 1
                        # x_text = "x: " + str(tvec[0])
                        # z_text = " z: " + str(tvec[2]) 
                        # cv2.putText(frame, x_text, (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                        # cv2.putText(frame, z_text, (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    if face2_num == 1 and face_flag == 0:
                        z_update = tvec[2, 0] - 150 # land region is 150 cm away from aruco-3
                        y_update = -(tvec[1, 0] + 20)
                        x_update = tvec[0, 0]
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = y_pid.update(y_update, sleep=0)
                        x_update = x_pid.update(x_update, sleep=0)

                        R, _ = cv2.Rodrigues(rvec)
                        V = np.matmul(R, [0, 0, 1])
                        rad = math.atan(V[0]/V[2])
                        deg = rad / math.pi * 180
                        yaw_update = yaw_pid.update(deg, sleep=0)

                        if abs(x_update) <= 10 and abs(z_update) <= 10 and abs(yaw_update) <= 2:
                            face_flag = 1

                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = int(mss(x_update))
                            yaw_update = int(mss(yaw_update))
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                    elif face2_num == 1 and face_flag == 1:
                        drone.send_rc_control(3,0,0,0)
                    
                    elif face2_num == 2: 
                        drone.send_rc_control(0, 0, 0, 0)

                        middle = int((faces_x[0][0]+ faces_x[1][0]) / 2)
                        x_update = abs(middle+15)
                        x_update = x_pid.update(x_update, sleep=0)
                        # line_x = abs(middle+15) + int(frame.shape[1] / 2)
                        print(f"Detect 2 faces! middle: {middle}")

                        # cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)  # (255, 0, 0) is the color (BGR format), 2 is the line thickness
                        if abs(x_update) < 2:
                            flag = 4
                            drone.send_rc_control(0,0,0,0)
                            time.sleep(0.5)
                            # drone.move_forward(200)
                            drone.send_rc_control(0, 50, 0, 0)
                            time.sleep(4)
                            drone.rotate_counter_clockwise(180)
                            time.sleep(2)
                            # drone.rotate_counter_clockwise(90)
                            # time.sleep(1)

                        else:
                            z_update = 0
                            y_update = 0
                            x_update = int(mss(x_update)) if int(mss(x_update)) <= 5 else 5
                            yaw_update = 0
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                        
                    else:
                        if face2_num > 2:
                            drone.send_rc_control(0, 0, 0, 0)  
                        else:
                            drone.send_rc_control(5,0,0,0)
                elif flag == 4:
                    drone.send_rc_control(5, 0,0,0)
                else:
                    drone.send_rc_control(0, 0,0,0)
            
            cv2.imshow('frame', frame)
            # cv2.waitKey(1)
            
if __name__ == '__main__':
    main()
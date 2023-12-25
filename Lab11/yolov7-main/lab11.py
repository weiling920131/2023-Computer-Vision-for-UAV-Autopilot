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



def line_follower(frame, h, w):
    threshold = 0.35
    num_rows = 3
    num_cols = 3
    cell_height = h // num_rows
    cell_width = w // num_cols
    black = np.zeros((num_cols, num_rows))

    for i in range(num_rows):
        for j in range(num_cols):
            cell = frame[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

            black_pixels = np.sum(cell == 0)
            pixels = cell_height * cell_width

            if(black_pixels > pixels * threshold):
                black[i][j] = 1

            cv2.rectangle(frame, (j * cell_width, i * cell_height), ((j + 1) * cell_width, (i + 1) * cell_height), 0, 2)

    return black.tolist()

WEIGHT = './runs/train/yolov7-lab09/weights/best.pt'
# WEIGHT = './runs/train/yolov7-lab09/weights/last.pt'
# WEIGHT = './yolov7-tiny.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"

model = attempt_load(WEIGHT, map_location=device)
if device == "cuda":
    model = model.half().to(device)
else:
    model = model.float().to(device)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

def mss(update, max_speed_threshold=30):
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

    flag = 33
    path = 0
    path_flag = 0
    doll = ''
    handle = False

    face_objectPoints = np.array([[0,0,0],[13,0,0],[0,17,0],[13,17,0]], dtype=np.float32)
    ped_objectPoints = np.array([[0,0,0],[0.5,0,0],[0,1.75,0],[0.4,1.61,0]], dtype=np.float32)
    while True:
        print("path_flag))))))))) ",path_flag)
        print("\n\nflag===========================", flag,'\n\n')
        print("doll++++++++++++++",doll)
        print("path:",path)
        
        key = cv2.waitKey(1)
        if key != -1:
            if key == ord('m'):
                handle = not handle
            elif key == ord('q'):
                break
            keyboard(drone, key)
        else:
            if handle:
                drone.send_rc_control(0, 0, 0, 0)
                continue
            
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
                        # if flag == 1 :
                        #     flag = 2
                            
                        z_update = tvec[i, 0, 2] - 70 # land region is 150 cm away from aruco-3
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
                        # congratulation, you finish final!
                        if abs(x_update) <= 10 and abs(z_update) <= 10 and abs(yaw_update) <= 2 and flag == 1:
                            # TODO: put detect result to doll
                            # doll = "Melody"
                            flag = 2    
                            if doll == "melody":
                                path = 1
                            elif doll == "cana":
                                path = 2

                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = int(mss(x_update))
                            yaw_update = int(mss(yaw_update)+3)
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                        break
                    
                    if id == 2 and (flag == 2 or flag == 3):
                        flag = 3
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
                        if abs(x_update) <= 10 and abs(z_update) <= 10 and abs(yaw_update) <= 2:
                            # TODO: put detect result to doll
                            drone.send_rc_control(0, 0,0, 0)
                            time.sleep(0.5)
                            drone.rotate_clockwise(90)
                            time.sleep(0.5)
                            drone.send_rc_control(0,0,0,0)
                            time.sleep(0.5)
                            drone.send_rc_control(0,-25,0,0)
                            time.sleep(2)
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
                        if flag == 4: # if not detect aruco-id-3, then turn left
                            drone.send_rc_control(0, 0, 0, -10)
                        elif flag == 2:
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

                            black = line_follower(frame, h, w)

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

                                elif path_flag == 4 and (black in corner_ulr or black in horizontal):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(-speed, 0, 0, 0)
                                    time.sleep(0.5)
                                    path_flag += 1

                                elif path_flag == 5 and (black in corner_dr or black in vertical):
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
                                        time.sleep(2)
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

                                elif path_flag == 1 and (black in corner_dr):
                                    drone.send_rc_control(0, 0, 0, 0)
                                    time.sleep(1)
                                    drone.send_rc_control(speed, 0, 0, 0)
                                    time.sleep(0.5)
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
                                    drone.send_rc_control(speed, 0, 0, 0)
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
                                        time.sleep(2)
                                    else:
                                        if black == [[0, 0, 0],
                                                    [0, 0, 0], 
                                                    [0, 0, 0]]:
                                            # print("stop")
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

                    black = line_follower(frame, h, w)

                    speed = 10
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

                        elif path_flag == 4 and (black in corner_ulr or black in horizontal):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(-speed, 0, 0, 0)
                            time.sleep(0.5)
                            path_flag += 1

                        elif path_flag == 5 and (black in corner_dr or black in vertical):
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

                        elif path_flag == 1 and (black in corner_dr):
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(speed, 0, 0, 0)
                            time.sleep(0.5)
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
                            drone.send_rc_control(speed, 0, 0, 0)
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

                elif flag == 33:
                    ####################
                    # face detect
                    ####################
                    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    rects = face_cascade.detectMultiScale(frame, 1.15, 6, minSize=(50,50))
                    face2_num = 0
                    faces_x = []
                    for x,y,w,h in rects:
                        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                        retval, rvec, tvec = cv2.solvePnP(face_objectPoints, np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]], dtype=np.float32), intrinsic, distortion)
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
                        if abs(x_update) < 5:
                            #TODO go to flag 4
                            flag = 4
                            # print("check")
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
                        break
                    else:
                        # if not found 2 faces yet, keep go to right
                        drone.send_rc_control(18, -1, 0, 0)  
                elif flag == 4:
                    drone.send_rc_control(0, 0,0,-10)
                else:
                    drone.send_rc_control(0, 0,0,0)
            
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            
if __name__ == '__main__':
    main()
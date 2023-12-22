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

corner1 = [
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

corner2 = [
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

corner3 = corner1

corner4 = [
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

corner5 = corner2

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
    #time.sleep(10)
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
    while True:
        print(flag)
        key = cv2.waitKey(1)
        if key != -1:
            if key == ord('q'):
                break
            keyboard(drone, key)
        else:
            frame = frame_read.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
            if markerIds is not None:
                frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
                rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)

                for i in range(len(markerIds)):
                    id = markerIds[i][0]
                    if id == 3 and flag == 0: 
                        z_update = tvec[i, 0, 2] - 50
                        y_update = -(tvec[i, 0, 1] + 3)
                        x_update = tvec[i, 0, 0]
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = y_pid.update(y_update, sleep=0)
                        x_update = x_pid.update(x_update, sleep=0)

                        R, _ = cv2.Rodrigues(rvec[i])
                        V = np.matmul(R, [0, 0, 1])
                        rad = math.atan(V[0]/V[2])
                        deg = rad / math.pi * 180
                        yaw_update = yaw_pid.update(deg, sleep=0)
                        print(yaw_update)

                        if abs(z_update) <= 5 and abs(yaw_update) <= 1 and abs(y_update) <= 10:
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(10, 0, 0, 0)
                            time.sleep(0.7)
                            flag = 1
                            x_direction = 1
                            y_direction = 0

                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = int(mss(x_update))
                            yaw_update = int(mss(yaw_update))
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)

                        break

                    elif id == 3 and flag == 6:
                        drone.land()
                    else:
                        drone.send_rc_control(10, 0, 0, 0)

            else:
                if flag > 0:
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
                    print(black, '\n')
                    if black == [[1, 1, 1],
                                [1, 1, 1], 
                                [1, 1, 1]]:
                        # print("111back")
                        # drone.send_rc_control(0, 0, 0, 0)
                        drone.send_rc_control(0, -speed, 0, 0)

                    elif flag == 1 and (black in corner1 or black in vertical):
                        drone.send_rc_control(0, 0, 0, 0)
                        time.sleep(1)
                        drone.send_rc_control(0, 0, speed, 0)
                        time.sleep(0.5)
                        flag = 2

                    elif flag == 2 and (black in corner2 or black in horizontal):
                        drone.send_rc_control(0, 0, 0, 0)
                        time.sleep(1)
                        drone.send_rc_control(speed, 0, 0, 0)
                        time.sleep(0.5)
                        flag = 3

                    elif flag == 3 and (black in corner3 or black in vertical):
                        drone.send_rc_control(0, 0, 0, 0)
                        time.sleep(1)
                        drone.send_rc_control(0, 0, speed, 0)
                        time.sleep(0.5)
                        flag = 4

                    elif flag == 4 and (black in corner4 or black in horizontal):
                        drone.send_rc_control(0, 0, 0, 0)
                        time.sleep(1)
                        drone.send_rc_control(-speed, 0, 0, 0)
                        time.sleep(0.5)
                        flag = 5

                    elif flag == 5 and (black in corner5 or black in vertical):
                        drone.send_rc_control(0, 0, 0, 0)
                        time.sleep(1)
                        flag = 6

                    elif flag == 6:
                        drone.send_rc_control(0, 0, -speed-10, 0)
                    else:                            
                        if flag == 1 or flag == 3: # 橫的 往右
                            # if [1, 1, 1] in black:
                            drone.send_rc_control(speed, 0, 0, 0)
                            # if black == [[1, 1, 1], [0, 0, 0], [0, 0, 0]]:
                            #     drone.send_rc_control(0, 0, speed, 0)       # u
                            # elif black == [[0, 0, 0], [0, 0, 0], [1, 1, 1]]:
                            #     drone.send_rc_control(0, 0, -speed, 0)      # d
                            # else:
                            #     drone.send_rc_control(speed, 0, 0, 0)       # r: gogo
                    
                        elif flag == 2 or flag == 4:   # 直的 往上
                            # if [1, 1, 1] in list(map(list, zip(*black))):
                            drone.send_rc_control(0, 0, speed+10, 0)
                            # if black == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]:
                            #     drone.send_rc_control(-speed, 0, 0, 0)      # l
                            # elif black == [[0, 0, 1], [0, 0, 1], [0, 0, 1]]:
                            #     drone.send_rc_control(speed, 0, 0, 0)       # r
                            # else:
                            #     drone.send_rc_control(0, 0, speed, 0)       # u: gogo

                        elif flag == 5: # 橫的 往左
                            # if [1, 1, 1] in black:
                            drone.send_rc_control(-speed, 0, 0, 0)
                            # if black == [[1, 1, 1], [0, 0, 0], [0, 0, 0]]:
                            #     drone.send_rc_control(0, 0, speed, 0)       # u
                            # elif black == [[0, 0, 0], [0, 0, 0], [1, 1, 1]]:
                            #     drone.send_rc_control(0, 0, -speed, 0)      # d
                            # else:
                            #     drone.send_rc_control(-speed, 0, 0, 0)      # l: gogo
                                
                        else:
                            if black == [[0, 0, 0],
                                        [0, 0, 0], 
                                        [0, 0, 0]]:
                                # print("stop")
                                drone.send_rc_control(0, 0, 0, 0)

                else:
                    drone.send_rc_control(0, 0, 0, 0)
                        

        cv2.imshow("drone", frame)

if __name__ == '__main__':
    main()
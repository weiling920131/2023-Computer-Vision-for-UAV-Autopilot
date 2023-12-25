import cv2 
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
# if go horizontol, cut put frame.height, if go vertical, cut put frame.width
def line_follower_horizontal(frame, height, middle):
    threshold = 0.2
    one, two, three = 0, 0, 0
    for i in range(height):
        for j in range(middle-3, middle):
            if frame[i, j] == 0: # pixel is black
                if(i < height/3):
                    one += 1
                elif(i < height/3*2):
                    two += 1
                else:
                    three += 1
                    
    # for i in range(cut):
    #     for j in range(non_cut):
    #         if frame[i, j] == 0:
    #             if(i < cut/3):
    #                 one += 1
    #             elif(i < cut/3*2):
    #                 two += 1
    #             else:
    #                 three += 1
    if one > height * 3 * threshold * 0.333:
        one = 1
    else:
        one = 0
    if two > height * 3 * threshold * 0.333:
        two = 1
    else:
        two = 0
    if three > height * 3 * threshold * 0.333:
        three = 1
    else:
        three = 0

    return one, two, three
# 
def line_follower_vertical(frame, width, middle):
    threshold = 0.2
    one, two, three = 0, 0, 0
    for i in range(width):
        for j in range(middle-3, middle):
            if frame[j, i] == 0: # pixel is black
                if(i < width/3):
                    one += 1
                elif(i < width/3*2):
                    two += 1
                else:
                    three += 1
                    
    # for i in range(cut):
    #     for j in range(non_cut):
    #         if frame[i, j] == 0:
    #             if(i < cut/3):
    #                 one += 1
    #             elif(i < cut/3*2):
    #                 two += 1
    #             else:
    #                 three += 1
    if one > width * 3 * threshold * 0.333:
        one = 1
    else:
        one = 0
    if two > width * 3 * threshold * 0.333:
        two = 1
    else:
        two = 0
    if three > width * 3 * threshold * 0.333:
        three = 1
    else:
        three = 0

    return one, two, three

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
    direction = 0 # dir = 1: horizontal, dir = -1: vertical
    constant_hor = 0
    constant_ver = 0
    corner = 0
    hor_pos = [0, 2]
    hor_neg = [4]
    ver_pos = [1, 3]
    ver_neg = [5]
    negative = -1
    while True:
        print("check")
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
                    if id == 4 and flag == 0:
                        # when id1 img < 150，then  
                        z_update = tvec[i, 0, 2] - 45
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

                        if abs(z_update) <= 5 and abs(yaw_update) <= 5 and abs(y_update) <= 10 and abs(x_update) <= 10:
                            drone.send_rc_control(20, 0, 0, 0)
                            time.sleep(1)
                            flag = 1
                            constant_hor = 1
                            constant_ver = 1
                            direction = 1
                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = int(mss(x_update))
                            yaw_update = int(mss(yaw_update))
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                    elif id == 4 and flag == 1:
                        drone.land()
            else:
                if flag == 0:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                _, frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                height, width = frame.shape
                
                if corner in hor_pos:
                    middle = int(width / 2) 
                    # go horizontal
                    one, two, three = line_follower_horizontal(frame, height, middle)
                    # direction = 1
                    # go to send_rc_control(x, z, y, yaw)
                    if one == 0 and two == 1 and three == 0: # straight right
                        drone.send_rc_control(5, 0, 0, 0)
                    elif one == 1 and two == 1 and three == 0: # slightly up
                        drone.send_rc_control(5, 0, 2.5, 0)
                    elif one == 0 and two == 1 and three == 1: # slightly down
                        drone.send_rc_control(5, 0, -2.5, 0)
                    elif one == 1 and two == 0 and three == 0: # up more
                        drone.send_rc_control(5, 0, 5, 0)
                    elif one == 0 and two == 0 and three == 1: # down more
                        drone.send_rc_control(5, 0, -5, 0)
                    elif one == 0 and two == 0 and three == 0:
                        drone.send_rc_control(0, 0, 0, 0)
                        # direction = -1
                        corner += 1
                        # constant_hor *= -1
                        continue
                elif corner in hor_neg:
                    middle = int(width / 2) 
                    # go horizontal
                    one, two, three = line_follower_horizontal(frame, height, middle)
                    # direction = 1
                    # go to send_rc_control(x, z, y, yaw)
                    if one == 0 and two == 1 and three == 0: # straight right
                        drone.send_rc_control(5*negative, 0, 0, 0)
                    elif one == 1 and two == 1 and three == 0: # slightly up
                        drone.send_rc_control(5*negative, 0, 2.5, 0)
                    elif one == 0 and two == 1 and three == 1: # slightly down
                        drone.send_rc_control(5*negative, 0, -2.5, 0)
                    elif one == 1 and two == 0 and three == 0: # up more
                        drone.send_rc_control(5*negative, 0, 5, 0)
                    elif one == 0 and two == 0 and three == 1: # down more
                        drone.send_rc_control(5*negative, 0, -5, 0)
                    elif one == 0 and two == 0 and three == 0:
                        drone.send_rc_control(0, 0, 0, 0)
                        # direction = -1
                        corner += 1
                        # constant_hor *= -1
                        continue
                elif corner in ver_pos:
                    middle = int(height / 2) 
                    # go horizontal
                    one, two, three = line_follower_vertical(frame, width, middle)
                    # go to send_rc_control(x, z, y, yaw)
                    if one == 0 and two == 1 and three == 0: # straight up
                        drone.send_rc_control(0, 0, 5, 0)
                    elif one == 1 and two == 1 and three == 0: # slightly right
                        drone.send_rc_control(2.5, 0, 5, 0)
                    elif one == 0 and two == 1 and three == 1: # slightly left
                        drone.send_rc_control(-2.5, 0, 5, 0)
                    elif one == 1 and two == 0 and three == 0: # right more
                        drone.send_rc_control(5, 0, 5, 0)
                    elif one == 0 and two == 0 and three == 1: # left more
                        drone.send_rc_control(-5, 0, 5, 0)
                    elif one == 0 and two == 0 and three == 0: # stop
                        drone.send_rc_control(0, 0, 0, 0)
                        # direction = 1
                        corner += 1
                        break
                elif corner in ver_neg:
                    middle = int(height / 2) 
                    # go horizontal
                    one, two, three = line_follower_vertical(frame, width, middle)
                    # go to send_rc_control(x, z, y, yaw)
                    if one == 0 and two == 1 and three == 0: # straight down
                        drone.send_rc_control(0, 0, 5*negative, 0)
                    elif one == 1 and two == 1 and three == 0: # slightly right
                        drone.send_rc_control(2.5, 0, 5*negative, 0)
                    elif one == 0 and two == 1 and three == 1: # slightly left
                        drone.send_rc_control(-2.5, 0, 5*negative, 0)
                    elif one == 1 and two == 0 and three == 0: # right more
                        drone.send_rc_control(5, 0, 5*negative, 0)
                    elif one == 0 and two == 0 and three == 1: # left more
                        drone.send_rc_control(-5, 0, 5*negative, 0)
                    elif one == 0 and two == 0 and three == 0: # stop
                        drone.send_rc_control(0, 0, 0, 0)
                        # direction = 1
                        corner += 1
                        break
                
                # if direction == 1: # go horizontal
                    
                # elif direction == -1: # go vertical
                #     # TODO 
                #     middle = int(height / 2) 
                #     # go horizontal
                #     one, two, three = line_follower_vertical(frame, width, middle)
                #     # go to send_rc_control(x, z, y, yaw)
                #     if one == 0 and two == 1 and three == 0: # straight right
                #         drone.send_rc_control(5, 0, 0, 0)
                #     elif one == 1 and two == 1 and three == 0: # slightly up
                #         drone.send_rc_control(5, 0, 2.5, 0)
                #     elif one == 0 and two == 1 and three == 1: # slightly down
                #         drone.send_rc_control(5, 0, -2.5, 0)
                #     elif one == 1 and two == 0 and three == 0: # up more
                #         drone.send_rc_control(5, 0, 5, 0)
                #     elif one == 0 and two == 0 and three == 1: # down more
                #         drone.send_rc_control(5, 0, -5, 0)
                #     elif one == 0 and two == 0 and three == 0:
                #         drone.send_rc_control(0, 0, 0, 0)
                #         direction = 1
                #         break
                    # else:
                    #     drone.send_rc_control(0, 0, 0, 0)
                    #     drone.land()
        cv2.imshow("drone", frame)

if __name__ == '__main__':
    main()
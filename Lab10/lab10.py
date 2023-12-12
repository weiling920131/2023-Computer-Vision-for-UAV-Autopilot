import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

def line_follower(frame, h, w, dir):
    threshold = 0.4
    line1 = [0, 0, 0]
    line2 = [0, 0, 0]
    if dir == 'x':
        for i in range(h):
            if frame[i, 0] == 0:
                if(i < h/3):
                    line1[0] += 1
                elif(i < h/3*2):
                    line1[1] += 1
                else:
                    line1[2] += 1

            if frame[i, w-1] == 0:
                if(i < h/3):
                    line2[0] += 1
                elif(i < h/3*2):
                    line2[1] += 1
                else:
                    line2[2] += 1

        for i in range(3):
            if line1[i] > h * threshold * 0.333:
                line1[i] = 1
            else:
                line1[i] = 0

            if line2[i] > h * threshold * 0.333:
                line2[i] = 1
            else:
                line2[i] = 0

    else:
        for j in range(w):
            if frame[0, j] == 0:
                if(j < w/3):
                    line1[0] += 1
                elif(j < w/3*2):
                    line1[1] += 1
                else:
                    line1[2] += 1

            if frame[h-1, j] == 0:
                if(j < w/3):
                    line2[0] += 1
                elif(j < w/3*2):
                    line2[1] += 1
                else:
                    line2[2] += 1

        for i in range(3):
            if line1[i] > w * threshold * 0.333:
                line1[i] = 1
            else:
                line1[i] = 0

            if line2[i] > w * threshold * 0.333:
                line2[i] = 1
            else:
                line2[i] = 0

    return line1, line2

# def swap_direction(frame, h, w, dir):
#     one, two, three = 0, 0, 0
#     if dir == 'x':
#         for i in range(h):
#             if frame[i, int(w/2)] == 0:
#                 if(i < h/3):
#                     one += 1
#                 elif(i < h/3*2):
#                     two += 1
#                 else:
#                     three += 1
#     else:
#         for j in range(w):
#             if frame[int(h/2), j] == 0:
#                 if(j < w/3):
#                     one += 1
#                 elif(j < w/3*2):
#                     two += 1
#                 else:
#                     three += 1

#     if one > three:
#         return 1
#     else:
#         return 3

def mss(update, max_speed_threshold=20):
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
                    if id == 4 and flag == 0: 
                        z_update = tvec[i, 0, 2] - 60
                        y_update = -(tvec[i, 0, 1] + 5)
                        x_update = tvec[i, 0, 0]
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = y_pid.update(y_update, sleep=0)
                        x_update = x_pid.update(x_update, sleep=0)
                        # print("y: ", y_update)

                        R, _ = cv2.Rodrigues(rvec[i])
                        V = np.matmul(R, [0, 0, 1])
                        rad = math.atan(V[0]/V[2])
                        deg = rad / math.pi * 180
                        yaw_update = yaw_pid.update(deg, sleep=0)

                        if abs(z_update) <= 5 and abs(yaw_update) <= 5 and abs(y_update) <= 10 and abs(x_update) <= 10:
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(2)
                            drone.send_rc_control(20, 0, 0, 0)
                            time.sleep(1)
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

                    elif id == 4 and flag == 2:
                        drone.land()
                    else:
                        drone.send_rc_control(10, 0, 0, 0)

            else:
                if flag == 1 or flag == 2:  
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)

                    _, frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    height, weight = frame.shape

                    x_line1, x_line2 = line_follower(frame, height, weight, 'x')
                    y_line1, y_line2 = line_follower(frame, height, weight, 'y')

                    # print(x_line1, x_line2)
                    # print(y_line1, y_line2)

                    if x_direction != 0 and (x_line2 == [0, 0, 0]):
                        drone.send_rc_control(0, 0, 0, 0)
                        time.sleep(2)
                        if sum(y_line1) > sum(y_line2):
                            x_direction = 0
                            y_direction = 1
                        else:
                            x_direction = 0
                            y_direction = -1

                    elif y_direction != 0 and (y_line1 == [0, 0, 0] or y_line2 == [0, 0, 0]):
                        drone.send_rc_control(0, 0, 0, 0)
                        time.sleep(2)
                        if sum(x_line1) > sum(x_line2):
                            x_direction = -1
                            y_direction = 0
                        else:
                            x_direction = 1
                            y_direction = 0
                    
                    # if (x_direction > 0 and y3 == 0) or (x_direction < 0 and y1 == 0):
                    #     drone.send_rc_control(0, 0, 0, 0)
                    #     time.sleep(2)
                    #     if swap_direction(frame, height, weight, 'x') == 1:
                    #         x_direction = 0
                    #         y_direction = 1
                    #     else:
                    #         x_direction = 0
                    #         y_direction = -1

                    # elif (y_direction > 0 and x1 == 0) or (y_direction < 0 and x3 == 0):
                    #     drone.send_rc_control(0, 0, 0, 0)
                    #     time.sleep(2)
                    #     if swap_direction(frame, height, weight, 'y') == 1:
                    #         x_direction = -1
                    #         y_direction = 0
                    #     else:
                    #         x_direction = 1
                    #         y_direction = 0
                    else:
                        flag = 2
                        if x_direction != 0:
                            drone.send_rc_control(20*x_direction, 0, 0, 0)

                            if x_line1[0] == 1 and x_line2[0] == 1:
                                drone.send_rc_control(0, 0, 5, 0)
                            elif x_line1[2] == 1 and x_line2[2] == 1:
                                drone.send_rc_control(0, 0, -5, 0)
                            # elif x_line1[1] == 1 and x_line2[1] == 1:
                            #     drone.send_rc_control(10*x_direction, 0, 0, 0)
                            # elif x_line1 == [1, 0, 0]:
                            #     drone.send_rc_control(10*x_direction, 0, 10, 0)
                            # elif x_line1 == [0, 0, 1]:
                            #     drone.send_rc_control(10*x_direction, 0, -10, 0)
                        else:
                            drone.send_rc_control(0, 0, 20*y_direction, 0)

                            if y_line1[0] == 1 and y_line2[0] == 1:
                                drone.send_rc_control(-5, 0, 0, 0)
                            elif y_line1[2] == 1 and y_line2[2] == 1:
                                drone.send_rc_control(5, 0, 0, 0)
                else:
                    drone.send_rc_control(0, 0, 0, 0)
                        

        cv2.imshow("drone", frame)

if __name__ == '__main__':
    main()
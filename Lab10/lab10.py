import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

exp = [
    [[0, 1, 0],
     [1, 1, 0], 
     [0, 0, 0]],

    [[0, 0, 0],
     [0, 1, 1], 
     [0, 1, 0]],

    [[0, 1, 0],
     [1, 1, 0], 
     [0, 0, 0]],

    [[1, 1, 0],
     [0, 1, 0], 
     [0, 0, 0]],
    
    [[0, 0, 0],
     [0, 1, 1], 
     [0, 1, 0]]
]

def line_follower(frame, h, w):
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

            if(black_pixels > pixels * 0.25):
                black[i][j] = 1

            cv2.rectangle(frame, (j * cell_width, i * cell_height), ((j + 1) * cell_width, (i + 1) * cell_height), 0, 2)

    return black

    # threshold = 0.4
    # line1 = 0
    # line2 = 0
    # # line1 = [0, 0, 0]
    # # line2 = [0, 0, 0]
    # if dir == 'x':
    #     for i in range(h):
    #         if frame[i, w/3] == 0:
    #             line1 += 1
    #         if frame[i, w/3*2] == 0:
    #             line2 += 1
        
    #     line1 = 1 if line1 > h * threshold else 0
    #     line2 = 1 if line2 > h * threshold else 0            

    #         # if frame[i, 0] == 0:
    #         #     if(i < h/3):
    #         #         line1[0] += 1
    #         #     elif(i < h/3*2):
    #         #         line1[1] += 1
    #         #     else:
    #         #         line1[2] += 1

    #         # if frame[i, w-1] == 0:
    #         #     if(i < h/3):
    #         #         line2[0] += 1
    #         #     elif(i < h/3*2):
    #         #         line2[1] += 1
    #         #     else:
    #         #         line2[2] += 1

    #     # for i in range(3):
    #     #     if line1[i] > h * threshold * 0.333:
    #     #         line1[i] = 1
    #     #     else:
    #     #         line1[i] = 0

    #     #     if line2[i] > h * threshold * 0.333:
    #     #         line2[i] = 1
    #     #     else:
    #     #         line2[i] = 0

    # else:
    #     for j in range(w):
    #         if frame[0, j] == 0:
    #             if(j < w/3):
    #                 line1[0] += 1
    #             elif(j < w/3*2):
    #                 line1[1] += 1
    #             else:
    #                 line1[2] += 1

    #         if frame[h-1, j] == 0:
    #             if(j < w/3):
    #                 line2[0] += 1
    #             elif(j < w/3*2):
    #                 line2[1] += 1
    #             else:
    #                 line2[2] += 1

    #     for i in range(3):
    #         if line1[i] > w * threshold * 0.333:
    #             line1[i] = 1
    #         else:
    #             line1[i] = 0

    #         if line2[i] > w * threshold * 0.333:
    #             line2[i] = 1
    #         else:
    #             line2[i] = 0

    # return line1, line2

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
                    if id == 2 and flag == 0: 
                        z_update = tvec[i, 0, 2] - 70
                        y_update = -(tvec[i, 0, 1])
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

                        if abs(z_update) <= 5 and abs(yaw_update) <= 2 and abs(y_update) <= 10:
                            drone.send_rc_control(0, 0, 0, 0)
                            time.sleep(1)
                            drone.send_rc_control(10, 0, 0, 0)
                            time.sleep(0.6)
                            flag = 1
                            x_direction = 1
                            y_direction = 0
                            # pass

                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = int(mss(x_update))
                            yaw_update = int(mss(yaw_update))
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)

                        break

                    elif id == 2 and flag == 6:
                        drone.land()
                    else:
                        drone.send_rc_control(10, 0, 0, 0)

            else:
                if flag > 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape
                    gray = gray[int(h/4):int(h/4*3), int(w/4):int(w/4*3)]
                    h, w = gray.shape
                    gray = cv2.GaussianBlur(gray, (7, 7), 0)

                    _, frame = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)

                    black = line_follower(frame, h, w)

                    print(black, '\n')

                    if(flag == 1):
                        if(np.array_equal(black, np.array([[0, 1, 0],
                                                            [1, 1, 0], 
                                                            [0, 0, 0]]))):
                            drone.send_rc_control(0, 0, 0, 0)
                            flag = 2

                        elif(np.array_equal(black, np.array([[1, 0, 0], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(-5, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[1, 1, 0], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(0, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[1, 1, 1], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(5, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[0, 1, 0], 
                                       [0, 1, 0], 
                                       [1, 1, 0]]))):
                            drone.send_rc_control(0, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 1], 
                                       [0, 0, 1], 
                                       [1, 1, 1]]))):
                            drone.send_rc_control(5, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(-5, 0, 0, 0)
                        
                        elif(np.array_equal(black, np.array([[1, 0, 0], 
                                       [1, 0, 0], 
                                       [1, 0, 0]]))):
                            drone.send_rc_control(-5, 0, 0, 0)

                        else:
                            drone.send_rc_control(10, 0, 0, 0)

                    if(flag == 2):
                        if(np.array_equal(black, np.array([[0, 0, 0],
                                        [0, 1, 1], 
                                        [0, 1, 0]]))):
                            drone.send_rc_control(0, 0, 0, 0)
                            flag = 3

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [0, 0, 1]]))):
                            drone.send_rc_control(5, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [0, 1, 1]]))):
                            drone.send_rc_control(0, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [1, 1, 1]]))):
                            drone.send_rc_control(-5, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[0, 1, 1], 
                                       [0, 1, 0], 
                                       [0, 1, 0]]))):
                            drone.send_rc_control(0, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[1, 1, 1], 
                                                          [1, 0, 0], 
                                                          [1, 0, 0]]))):
                            drone.send_rc_control(-5, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(0, 0, -5, 0)

                        else:
                            drone.send_rc_control(0, 0, 10, 0)

                    if(flag == 3):
                        if(np.array_equal(black, np.array([[0, 1, 0],
                                                            [1, 1, 0], 
                                                            [0, 0, 0]]))):
                            drone.send_rc_control(0, 0, 0, 0)
                            flag = 4

                        elif(np.array_equal(black, np.array([[1, 0, 0], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(-5, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[1, 1, 0], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(0, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[1, 1, 1], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(5, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[0, 1, 0], 
                                       [0, 1, 0], 
                                       [1, 1, 0]]))):
                            drone.send_rc_control(0, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 1], 
                                       [0, 0, 1], 
                                       [1, 1, 1]]))):
                            drone.send_rc_control(5, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(-5, 0, 0, 0)

                        else:
                            drone.send_rc_control(10, 0, 0, 0)

                    if(flag == 4):
                        if(np.array_equal(black, np.array([[1, 1, 0],
                                                            [0, 1, 0], 
                                                            [0, 0, 0]]))):
                            drone.send_rc_control(0, 0, 0, 0)
                            flag = 5

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                                        [0, 0, 0], 
                                                        [1, 0, 0]]))):
                            drone.send_rc_control(-5, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [1, 1, 0]]))):
                            drone.send_rc_control(0, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [1, 1, 1]]))):
                            drone.send_rc_control(5, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[1, 1, 0], 
                                       [0, 1, 0], 
                                       [0, 1, 0]]))):
                            drone.send_rc_control(0, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[1, 1, 1], 
                                                          [0, 0, 1], 
                                                          [0, 0, 1]]))):
                            drone.send_rc_control(5, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(0, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[1, 0, 0], 
                                       [1, 0, 0], 
                                       [1, 0, 0]]))):
                            drone.send_rc_control(-5, 0, 0, 0)

                        else:
                            drone.send_rc_control(0, 0, 10, 0)  

                    if(flag == 5):
                        if(np.array_equal(black, np.array([[0, 0, 0],
                                                        [0, 1, 1], 
                                                        [0, 1, 0]]))):
                            drone.send_rc_control(0, 0, 0, 0)
                            flag = 6

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [0, 0, 1]]))):
                            drone.send_rc_control(5, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [0, 1, 1]]))):
                            drone.send_rc_control(0, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [1, 1, 1]]))):
                            drone.send_rc_control(-5, 0, -5, 0)

                        elif(np.array_equal(black, np.array([[0, 1, 1], 
                                       [0, 1, 0], 
                                       [0, 1, 0]]))):
                            drone.send_rc_control(0, 0, 5, 0)

                        elif(np.array_equal(black, np.array([[1, 1, 1], 
                                                          [1, 0, 0], 
                                                          [1, 0, 0]]))):
                            drone.send_rc_control(-5, 0, 5, 0)
                        
                        elif(np.array_equal(black, np.array([[0, 0, 0], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(5, 0, 0, 0)

                        elif(np.array_equal(black, np.array([[1, 1, 1], 
                                       [0, 0, 0], 
                                       [0, 0, 0]]))):
                            drone.send_rc_control(0, 0, 5, 0)

                        else:
                            drone.send_rc_control(-10, 0, 0, 0)

                    if(flag == 6):
                        drone.send_rc_control(0, 0, -10, 0)




                        
                    
                    
                    # if (x_direction > 0 and black[1][2] == 0) or (x_direction < 0 and black[1][0] == 0):
                    #     if black[0][0] or black[0][1] or black[0][2]:
                    #         # drone.send_rc_control(10*x_direction, 0, 0, 0)
                    #         drone.send_rc_control(0, 0, 0, 0)
                    #         time.sleep(0.5)
                    #         drone.send_rc_control(0, 0, 10, 0)
                    #         # time.sleep(0.5)

                    #         x_direction = 0
                    #         y_direction = 1

                    #     elif black[2][0] or black[2][1] or black[2][2]:
                    #         # drone.send_rc_control(10*x_direction, 0, 0, 0)
                    #         drone.send_rc_control(0, 0, 0, 0)
                    #         time.sleep(0.5)
                    #         drone.send_rc_control(0, 0, -10, 0)
                    #         # time.sleep(0.5)

                    #         x_direction = 0
                    #         y_direction = -1

                    # elif (y_direction > 0 and black[0][1] == 0) or (y_direction < 0 and black[2][1] == 0):
                    #     if black[0][2] or black[1][2] or black[2][2]:
                    #         # drone.send_rc_control(0, 0, 10*y_direction, 0)
                    #         drone.send_rc_control(0, 0, 0, 0)
                    #         time.sleep(0.5)
                    #         drone.send_rc_control(10, 0, 0, 0)
                    #         # time.sleep(0.5)

                    #         x_direction = 1
                    #         y_direction = 0

                    #     elif black[0][0] or black[1][0] or black[2][0]:
                    #         # drone.send_rc_control(0, 0, 10*y_direction, 0)
                    #         drone.send_rc_control(0, 0, 0, 0)
                    #         time.sleep(0.5)
                    #         drone.send_rc_control(-10, 0, 0, 0)
                    #         # time.sleep(0.5)

                    #         x_direction = -1
                    #         y_direction = 0

                    
                    # print('x:', x_direction, 'y:', y_direction)
                    # flag = 2
                    # if x_direction != 0:
                    #     drone.send_rc_control(20*x_direction, -1, 0, 0)

                    #     if np.sum(black, axis=1)[0] > 1:
                    #         drone.send_rc_control(0, 0, 5, 0)
                    #     elif np.sum(black, axis=1)[2] > 1:
                    #         drone.send_rc_control(0, 0, -5, 0)

                    # elif y_direction != 0:
                    #     drone.send_rc_control(0, 0, 20*y_direction, 0)

                    #     if np.sum(black, axis=0)[0] > 1:
                    #         drone.send_rc_control(-5, 0, 0, 0)
                    #     elif np.sum(black, axis=0)[2] > 1:
                    #         drone.send_rc_control(5, 0, 0, 0)
                else:
                    drone.send_rc_control(0, 0, 0, 0)
                        

        cv2.imshow("drone", frame)

if __name__ == '__main__':
    main()
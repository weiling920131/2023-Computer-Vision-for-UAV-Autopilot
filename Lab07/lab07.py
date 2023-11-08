import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard


def mss(update, max_speed_threshold=50):
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

    flag = 1
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
                    if id == 1:
                        # when id1 img < 150，then  
                        z_update = tvec[i, 0, 2] - 60
                        y_update = -(tvec[i, 0, 1] + 20)
                        # x_update = tvec[i, 0, 0]
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = y_pid.update(y_update, sleep=0)
                        # x_update = x_pid.update(x_update, sleep=0)

                        R, _ = cv2.Rodrigues(rvec[i])
                        V = np.matmul(R, [0, 0, 1])
                        rad = math.atan(V[0]/V[2])
                        deg = rad / math.pi * 180
                        yaw_update = yaw_pid.update(deg, sleep=0)

                        if abs(z_update) <= 15 and abs(yaw_update) <= 5:
                            drone.send_rc_control(0, 0, 50, 0)
                            time.sleep(2)
                            drone.send_rc_control(0, 40, 0, 0)
                            time.sleep(1.5)
                            flag = 2

                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = 0
                            yaw_update = int(mss(yaw_update))
                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)

                        break

                    elif id == 2:
                        z_update = tvec[i, 0, 2] - 50
                        y_update = -(tvec[i, 0, 1] + 20)
                        # x_update = tvec[i, 0, 0]
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = y_pid.update(y_update, sleep=0)
                        # x_update = x_pid.update(x_update, sleep=0)

                        R, _ = cv2.Rodrigues(rvec[i])
                        V = np.matmul(R, [0, 0, 1])
                        rad = math.atan(V[0]/V[2])
                        deg = rad / math.pi * 180
                        yaw_update = yaw_pid.update(deg, sleep=0)

                        if abs(z_update) <= 15 and abs(yaw_update) <= 5 :
                            drone.send_rc_control(0, 0, -50, 0)
                            time.sleep(1.5)
                            drone.send_rc_control(0, 60, 0, 0)
                            time.sleep(2)
                            flag = 3
                            # drone.land()

                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = 0
                            yaw_update = int(mss(yaw_update))

                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                            # print(0, int(z_update//2), int(y_update), int(yaw_update))

                        break

                    elif id == 0:
                        z_update = tvec[i, 0, 2] - 50
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = -(tvec[i, 0, 1] + 20)
                        y_update = y_pid.update(y_update, sleep=0)
                        R, _ = cv2.Rodrigues(rvec[i])
                        V = np.matmul(R, [0, 0, 1])
                        rad = math.atan(V[0]/V[2])
                        deg = rad / math.pi * 180
                        yaw_update = yaw_pid.update(deg, sleep=0)

                        z_update = int(mss(z_update) // 2)
                        y_update = int(mss(y_update))
                        x_update = 0
                        yaw_update = int(mss(yaw_update))

                        drone.send_rc_control(x_update, z_update, y_update, yaw_update)

                    elif id == 3 and flag != 4:
                        z_update = tvec[i, 0, 2] - 50
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = -(tvec[i, 0, 1] + 20)
                        y_update = y_pid.update(y_update, sleep=0)
                        R, _ = cv2.Rodrigues(rvec[i])
                        V = np.matmul(R, [0, 0, 1])
                        rad = math.atan(V[0]/V[2])
                        deg = rad / math.pi * 180
                        yaw_update = yaw_pid.update(deg, sleep=0)

                        if abs(z_update) <= 15 and abs(yaw_update) <= 5 :
                            flag = 4
                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = 0
                            yaw_update = int(mss(yaw_update))

                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                    elif id == 4 and flag != 5:
                        z_update = tvec[i, 0, 2] - 50
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = -(tvec[i, 0, 1] + 20)
                        y_update = y_pid.update(y_update, sleep=0)

                        x_update = tvec[i, 0, 0]
                        x_update = x_pid.update(x_update, sleep=0)

                        R, _ = cv2.Rodrigues(rvec[i])
                        V = np.matmul(R, [0, 0, 1])
                        rad = math.atan(V[0]/V[2])
                        deg = rad / math.pi * 180
                        yaw_update = yaw_pid.update(deg, sleep=0)

                        if abs(z_update) <= 15 and abs(yaw_update) <= 5 and abs(x_update) <= 5:
                            flag = 5
                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = 0
                            yaw_update = int(mss(yaw_update))

                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                    elif id == 5 and flag !=6:
                        z_update = tvec[i, 0, 2] - 150
                        z_update = z_pid.update(z_update, sleep=0)
                        y_update = -(tvec[i, 0, 1] + 20)
                        y_update = y_pid.update(y_update, sleep=0)

                        x_update = tvec[i, 0, 0]
                        x_update = x_pid.update(x_update, sleep=0)

                        R, _ = cv2.Rodrigues(rvec[i])
                        V = np.matmul(R, [0, 0, 1])
                        rad = math.atan(V[0]/V[2])
                        deg = rad / math.pi * 180
                        yaw_update = yaw_pid.update(deg, sleep=0)

                        if abs(z_update) <= 5 and abs(yaw_update) <= 5 and abs(x_update) <= 5:
                            flag = 6
                        else:
                            z_update = int(mss(z_update) // 2)
                            y_update = int(mss(y_update))
                            x_update = 0
                            yaw_update = int(mss(yaw_update))

                            drone.send_rc_control(x_update, z_update, y_update, yaw_update)
                    else:
                        if flag == 4:
                            drone.send_rc_control(0, 0, 0, 30)
                        elif flag == 5:
                            drone.send_rc_control(-50, 0, 0, 0)
                        elif flag == 6:
                            drone.send_rc_control(0, 0, -30, 0)

            else:
                if flag == 2:
                    drone.send_rc_control(0, 0, -50, 0)
                elif flag == 4:
                    drone.send_rc_control(0, 0, 0, 30)
                elif flag == 5:
                    drone.send_rc_control(-50, 0, 0, 0)
                elif flag == 6:
                    drone.send_rc_control(0, 0, -30, 0)
                else:
                    drone.send_rc_control(0, 0, 0, 0)
        
        cv2.imshow("drone", frame)
    
    #cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

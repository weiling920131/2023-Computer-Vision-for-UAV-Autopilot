import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard


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
    yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

    z_pid.initialize()
    y_pid.initialize()
    yaw_pid.initialize()


    while True:
        key = cv2.waitKey(1)
        if key != -1:
            keyboard(drone, key)
        else:
            frame = frame_read.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
            if markerIds is not None:
                frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
                rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)

                for i in range(len(markerIds)):
                    if markerIds[i][0] != 0:
                        continue
                    # print(rvec[i])
                    # print(tvec, '\n')

                    z_update = tvec[i, 0, 2] - 100
                    # print("org_z: ", str(z_update))
                    z_update = z_pid.update(z_update, sleep=0)
                    # print("pid_z: ", str(z_update))
                    y_update = -(tvec[i, 0, 1] + 20)
                    # print("org_z: ", str(z_update))
                    y_update = y_pid.update(y_update, sleep=0)
                    # print("pid_z: ", str(z_update))
                    R, _ = cv2.Rodrigues(rvec[i])
                    V = np.matmul(R, [0, 0, 1])
                    rad = math.atan(V[0]/V[2])
                    deg = rad / math.pi * 180
                    print(deg)
                    yaw_update = yaw_pid.update(deg, sleep=0)

                    z_update = mss(z_update)
                    y_update = mss(y_update)
                    yaw_update = mss(yaw_update)
                    print(z_update, y_update, yaw_update)

                    drone.send_rc_control(0, int(z_update//2), int(y_update), int(yaw_update))

                    frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec[0], tvec[0], 0.1)
                    text = " z: " + str(tvec[0, 0, 2])
                    cv2.putText(frame, text, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                drone.send_rc_control(0, 0, 0, 0)
        
        cv2.imshow("drone", frame)
    
    #cv2.destroyAllWindows()



if __name__ == '__main__':
    main()


import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

def threshold(x, threshold=50):
    if (x > 50):
def keyboard(self, key):
    #global is_flying
    print("key:", key)
    fb_speed = 40
    lf_speed = 40
    ud_speed = 50
    degree = 30
    if key == ord('1'):
        self.takeoff()
        #is_flying = True
    if key == ord('2'):
        self.land()
        #is_flying = False
    if key == ord('3'):
        self.send_rc_control(0, 0, 0, 0)
        print("stop!!!!")
    if key == ord('w'):
        self.send_rc_control(0, fb_speed, 0, 0)
        print("forward!!!!")
    if key == ord('s'):
        self.send_rc_control(0, (-1) * fb_speed, 0, 0)
        print("backward!!!!")
    if key == ord('a'):
        self.send_rc_control((-1) * lf_speed, 0, 0, 0)
        print("left!!!!")
    if key == ord('d'):
        self.send_rc_control(lf_speed, 0, 0, 0)
        print("right!!!!")
    if key == ord('z'):
        self.send_rc_control(0, 0, ud_speed, 0)
        print("down!!!!")
    if key == ord('x'):
        self.send_rc_control(0, 0, (-1) *ud_speed, 0)
        print("up!!!!")
    if key == ord('c'):
        self.send_rc_control(0, 0, 0, degree)
        print("rotate!!!!")
    if key == ord('v'):
        self.send_rc_control(0, 0, 0, (-1) *degree)
        print("counter rotate!!!!")
    if key == ord('5'):
        height = self.get_height()
        print(height)
    if key == ord('6'):
        battery = self.get_battery()
        print (battery)

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
    x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

    x_pid.initialize()
    y_pid.initialize()
    z_pid.initialize()
    yaw_pid.initialize()
    
    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        print(f'markerIds: {markerIds}')
        if markerIds is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
            print(rvec)
            print(tvec, '\n')
            # for i in range(len(markerIds)):
            for i in range(rvec.shape[0]):
                id = markerIds[i][0]
                # marco id = 0 
                if id != 0:
                    continue
                rotM = np.zeros(shape=(3, 3))
                # this is use to transfer vector rvec[i] into matrix 
                cv2.Rodrigues(rvec[i], rotM)

            drone.send_rc_control(0, int(z_update//2), int(y_update), int(yaw_update))
            frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec[0], tvec[0], 0.1)
            text = "x:" + str(tvec[0, 0, 0]) + " y:" + str(tvec[0, 0, 1]) + " z: " + str(tvec[0, 0, 2])
            cv2.putText(frame, text, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("drone", frame)
        key = cv2.waitKey(200)
        if key != -1:
            keyboard(drone, key)
    
    #cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

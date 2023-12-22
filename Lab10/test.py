import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

def line_follower(frame, h, w, dir):
    num_rows = 3
    num_cols = 3
    cell_height = h // num_rows
    cell_width = w // num_cols
    black = np.zeros((num_cols, num_rows))

    # 遍歷每個九宮格
    for i in range(num_rows):
        for j in range(num_cols):
            # 提取當前格子的區域
            cell = frame[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            # print(i * cell_height, (i + 1) * cell_height)
            # print(j * cell_width, (j + 1) * cell_width)

            # 計算當前格子中黑色像素的數量
            black_pixels = np.sum(cell == 0)
            pixels = cell_height * cell_width

            # 輸出結果
            # print(f"Cell ({i + 1}, {j + 1}): {black_pixels} / {pixels} = {black_pixels / pixels}")
            if(black_pixels > pixels * 0.2):
                black[i][j] = 1

            # 在畫面上劃出九宮格的邊界
            cv2.rectangle(frame, (j * cell_width, i * cell_height), ((j + 1) * cell_width, (i + 1) * cell_height), 0, 2)

    return black, frame

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

    while True:
        key = cv2.waitKey(1)
        if key != -1:
            if key == ord('q'):
                break
            keyboard(drone, key)
        else:
            frame = frame_read.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            gray = gray[int(h/4):int(h/4*3), int(w/4):int(w/4*3)]
            h, w = gray.shape
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            _, frame = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)

            black, frame = line_follower(frame, h, w, 0)

            print(black, '\n')
                        

        cv2.imshow("drone", frame)

if __name__ == '__main__':
    main()
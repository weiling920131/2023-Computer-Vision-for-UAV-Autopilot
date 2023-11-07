import cv2
import numpy as np
cap = cv2.VideoCapture(1)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
object_points = []
image_points = []
# image_size = 
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    pattern_size = (9, 6)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(frame, pattern_size, None)
    image_size = gray.shape[::-1]
    winSize = (11, 11)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    if ret:
        object_points.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, winSize, zeroZone, criteria)
        image_points.append(corners)
        # cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
    cv2.waitKey(50)
    if (len(image_points) >= 40):
        break

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
fs = cv2.FileStorage('camera_calibration.xml', cv2.FILE_STORAGE_WRITE)
fs.write("intrinsic", camera_matrix)
fs.write("distortion", dist_coeffs)
fs.release()
    
cap.release()
cv2.destroyAllWindows()
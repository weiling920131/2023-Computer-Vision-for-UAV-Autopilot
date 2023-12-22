import cv2
import numpy as np

def IsInRegion(corner, x, y):
    at = float((corner[1,1]-corner[0,1])/(corner[1,0]-corner[0,0]))
    bt = float(corner[0,1] - at*corner[0,0])
    ad = float((corner[3,1]-corner[2,1])/(corner[3,0]-corner[2,0]))
    bd = float(corner[2,1] - ad*corner[2,0])
    ar = float((corner[1,1]-corner[3,1])/(corner[1,0]-corner[3,0]))
    br = float(corner[3,1] - ar*corner[3,0])
    al = float((corner[2,1]-corner[0,1])/(corner[2,0]-corner[0,0]))
    bl = float(corner[2,1] - al*corner[2,0])

    if y > at*x+bt and y < ad*x+bd and y > al*x+bl and y > ar*x+br:
        return True
    else:
        return False
    
cap = cv2.VideoCapture(0)
screen = cv2.imread('screen.jpg')
screen_corner = np.array([[275, 189],[618, 87],[262, 403],[628, 370]])

while(1):
    ret, img = cap.read()
    if not ret: break
    h, w, _ = img.shape

    mat3 = cv2.getPerspectiveTransform(np.array([[0, 0],[w-1, 0],[0, h-1],[w-1, h-1]], dtype=np.float32), np.array(screen_corner, dtype=np.float32))
    mat3 = mat3/mat3[2,2]

    img = np.vstack([img,np.zeros([1, w, 3])])
    img = np.hstack([img,np.zeros([h + 1, 1, 3])])
    for y in range(87, 404):
        for x in range(262, 629):
            # if not IsInRegion(screen_corner, x, y):
            #     continue
            
            point = np.array(np.dot(np.linalg.inv(mat3), np.array([x, y, 1])),dtype=np.float32)
            i, j = point[0:2]/point[2]
            if i >= w or j >= h or i < 0 or j < 0:
                continue

            x1 = int(np.floor(i))
            x2 = x1 + 1
            y1 = int(np.floor(j)) 
            y2 = y1 + 1

            top = img[y1, x1]*(y2 - j) + img[y1, x2]*(j - y1)
            bottom = img[y2, x1]*(y2 - j) + img[y2, x2]*(j - y1)

            screen[y, x] = top*(x2 - i) + bottom*(i - x1)
    cv2.imshow('screen', screen)
    cv2.waitKey(33)
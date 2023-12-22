import cv2
import numpy as np

img = cv2.imread('input/test.jpg')
rect = cv2.selectROI('img', img, True, False)

b_Model = np.zeros((1,65),np.float64)
f_Model = np.zeros((1,65),np.float64)
iter_num = 15

mask_new, b_model, f_model=cv2.grabCut(img, None, rect, b_Model, f_Model, iter_num, 
                                        cv2.GC_INIT_WITH_RECT) 

mask = np.where((mask_new==0)|(mask_new==2),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]

cv2.imwrite('output/GrabCut.png', img)
cv2.imshow('img', img)
cv2.waitKey(0)
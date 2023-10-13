import cv2
import numpy as np

img = cv2.imread('images/test.jpg')
rect = cv2.selectROI('roi', img)

iter_num = 15

b_Model = np.zeros((1,65), np.float64)
f_Model = np.zeros((1,65), np.float64)
mask_new, b_model, f_model = cv2.grabCut(img, None, rect, b_Model, f_Model, iter_num, cv2.GC_INIT_WITH_RECT) 
mask = np.where((mask_new == 0) | (mask_new == 2), 0, 1).astype('uint8')
img = img * mask[:, :, np.newaxis]

cv2.imshow('1', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('output/1.jpg', img)
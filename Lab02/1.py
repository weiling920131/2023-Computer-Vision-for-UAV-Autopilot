import cv2
import numpy as np

# img = cv2.imread('images/12.png')

# img = cv2.imread(r"C:\Users\user\Downloads\2.jpg")
img = cv2.imread('images/filtering.jpg')
# img = cv2.imread('images/1.jpg')
# img = cv2.imread('images/1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5,5), 0)

x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
y_kernel = x_kernel.T
gx = cv2.filter2D(img, -1, x_kernel)
gy = cv2.filter2D(img, -1, y_kernel)

# new_img = np.array(gx + gy)
new_img = np.array(abs(gx) + abs(gy))
# new_img = np.around(np.sqrt(gx**2 + gy**2))
# new_img = np.sqrt(np.square(gx) + np.square(gy))
# new_img *= 255.0 / new_img.max()
cv2.imwrite('output/1.png', new_img.astype(np.float32))
# img = cv2.Sobel(img, -1, 1, 1, 7)
# cv2.imwrite('output/1.png', img.astype(np.float32))

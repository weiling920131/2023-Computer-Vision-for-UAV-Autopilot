import cv2
import numpy as np

cap = cv2.VideoCapture('images/train.mp4')
backSub = cv2.createBackgroundSubtractorMOG2()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/2.mp4', fourcc, 20.0, (320, 240))

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break

    fgmask = backSub.apply(frame)

    shadowval = backSub.getShadowValue()
    ret, nmask = cv2.threshold(fgmask, shadowval, 255, cv2.THRESH_BINARY)

    h, w = nmask.shape
    label = np.zeros((h, w), dtype=int)
    cur_label = 0
    equal = {}
    for i in range(h):
        for j in range(w):
            if nmask[i, j] == 255:
                neighbor = []
                if i != 0 and label[i-1, j] != 0:
                    neighbor += [label[i-1, j]]
                if j != 0 and label[i, j-1] != 0:
                    neighbor += [label[i, j-1]]

                n = len(neighbor)
                if n == 0:
                    cur_label += 1
                    label[i, j] = cur_label
                elif n == 1:
                    label[i, j] = neighbor[0]
                elif neighbor[0] != neighbor[1]:
                    label[i, j] = min(neighbor)
                    equal[neighbor[0]] = equal[neighbor[0]] + [neighbor[1]] if neighbor[0] in equal else [neighbor[1]]
                    equal[neighbor[1]] = equal[neighbor[1]] + [neighbor[0]] if neighbor[1] in equal else [neighbor[0]]

    # print(equal)      
    area = {}
    for i in range(h):
        for j in range(w):
            l = label[i, j]
            if l != 0:
                while l in equal:
                    if min(equal[l]) > l:
                        break
                    l = min(equal[l])
                label[i, j] = l
                area[l] = area[l] + [[i, j]] if l in area else [[i, j]]

    # print(area)
    T = 400
    for i in area:
        if len(area[i]) > T:
            a = np.array(area[i]).T
            h1 = min(a[0])
            w1 = min(a[1])
            h2 = max(a[0])
            w2 = max(a[1])
            cv2.rectangle(frame, (w1, h1), (w2, h2), (0, 0, 255), 3, cv2.LINE_AA)
    
    out.write(frame)
    # cv2.imshow("frame", frame)
    # cv2.waitKey(33)

cap.release()
out.release()
cv2.destroyAllWindows()
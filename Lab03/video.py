import cv2
import numpy as np
import os

def select_object(frame, threshold, resolve, areas, cc_mask):
    h, w, _ = frame.shape
    frame = np.array(frame, dtype=np.int32)
    
    for i in range(h):
        for j in range(w):
            if not cc_mask[i][j]:
                continue
            for set in resolve:
                if cc_mask[i][j] in set:
                    cc_mask[i][j] = min(set)
    tmp = dict()
    for i in range(h):
        for j in range(w):
            if cc_mask[i][j] in tmp:
                tmp[cc_mask[i][j]] += 1
            else:
                tmp[cc_mask[i][j]] = 1
    for label, area in tmp.items():
        if area < threshold or area > 10000:
            continue
        right = 0
        left = w
        top = 0
        bottom = h
        for i in range(h):
            for j in range(w):
                if label == cc_mask[i][j]:
                    right = max(right, j)
                    left = min(left, j)
                    top = max(top, i)
                    bottom = min(bottom, i)
        cv2.rectangle(frame, (right, top), (left, bottom), (0, 255, 0), 2)


    # for set in resolve:
    #     area = 0
    #     for label in set:
    #         area += areas[label]
        
    #     if area > threshold:
    #         right = 0
    #         left = w
    #         top = 0
    #         bottom = h
    #         for i in range(h):
    #             for j in range(w):
    #                 if cc_mask[i][j] in set:
    #                     right = max(right, j)
    #                     left = min(left, j)
    #                     top = max(top, i)
    #                     bottom = min(bottom, i)
    #         cv2.rectangle(frame, (right, top), (left, bottom), (0, 255, 0), 2)
            
    np.clip(frame, 0, 255, out=frame)
    frame = np.array(frame, dtype=np.uint8)
    return frame

        
def connected_component(nmask):
    h, w = nmask.shape
    resolve = []
    areas = dict()
    new_mask = np.zeros_like(nmask, dtype=np.int32)
    unused = 2

    for i in range(h):
        for j in range(w):
            if nmask[i][j] == 0: 
                continue

            if not i and not j:
                new_mask[i][j] = 1
                areas[1] = 1
            elif not i and j:
                if new_mask[i][j-1] != 0:
                    new_mask[i][j] = new_mask[i][j-1]
                    areas[new_mask[i][j]] += 1
                else:
                    new_mask[i][j] = unused
                    areas[unused] = 1
                    unused += 1
            elif not j and i:
                if new_mask[i-1][j] != 0:
                    new_mask[i][j] = new_mask[i-1][j]
                    areas[new_mask[i][j]] += 1
                else:
                    new_mask[i][j] = unused
                    areas[unused] = 1
                    unused += 1
            else:
                if new_mask[i-1][j] != 0 and new_mask[i][j-1] != 0:
                    new_mask[i][j] = min(new_mask[i-1][j], new_mask[i][j-1])
                    areas[new_mask[i][j]] += 1
                    check = True
                    for set in resolve:
                        if new_mask[i][j] in set:
                            set.add(max(new_mask[i-1][j], new_mask[i][j-1]))
                            check = False
                            break
                    if check:
                        resolve.append({new_mask[i-1][j], new_mask[i][j-1]})

                elif new_mask[i-1][j] != 0:
                    new_mask[i][j] = new_mask[i-1][j]
                    areas[new_mask[i][j]] += 1
                elif new_mask[i][j-1] != 0:
                    new_mask[i][j] = new_mask[i][j-1]
                    areas[new_mask[i][j]] += 1
                else:
                    new_mask[i][j] = unused
                    areas[unused] = 1
                    unused += 1
    return resolve, areas, new_mask


def solve(video, threshold):
    backSub = cv2.createBackgroundSubtractorMOG2()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/output.mp4', fourcc, 20.0, (320, 240))
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        fgmask = backSub.apply(frame) ##find foreground
        shadowval = backSub.getShadowValue() ## elminiate shadow
        ret, nmask = cv2.threshold(fgmask, shadowval, 255, cv2.THRESH_BINARY)
        if not ret:
            break
        x, y = nmask.shape
        if np.count_nonzero(nmask == 255) < x*y*0.1:

            resolve, areas, cc_mask = connected_component(nmask)
            selected_frame = select_object(frame, threshold, resolve, areas, cc_mask)
        else:
            selected_frame = frame
        out.write(selected_frame)
        # cv2.imshow("frame", selected_frame)
        # if cv2.waitKey(33) & 0xFF == ord(']'):
        #     break

if __name__ == '__main__':

    video = cv2.VideoCapture('input/train.mp4')
    
    if not video.isOpened():
        print('error : The video is not found.')
        exit(1)

    solve(video, threshold=450)
import cv2
import numpy as np
import os

def select_object(frame, threshold, resolve, areas, cc_mask):
    frame = np.array(frame, dtype=np.int32)
    for set in resolve:
        area = 0
        for label in set:
            area += areas[label]
        
        if area > threshold and area < 10000:
            x_array, y_array = np.where(np.isin(cc_mask, np.array(list(set)))) ##notice
            # print('\n\n')
            # print(x_array)
            # print(y_array)
            # print('\n\n')
            cv2.rectangle(frame, (x_array[0], y_array[0]), (int(x_array[0]+area**0.52), int(y_array[0]+area**0.48)), (0, 255, 0), 2)
            
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
            elif not i:
                if nmask[i][j-1] != 0:
                    new_mask[i][j] = new_mask[i][j-1]
                    areas[new_mask[i][j]] += 1
                else:
                    new_mask[i][j] = unused
                    areas[unused] = 1
                    unused += 1
            elif not j:
                if nmask[i-1][j] != 0:
                    new_mask[i][j] = new_mask[i-1][j]
                    areas[new_mask[i][j]] += 1
                else:
                    new_mask[i][j] = unused
                    areas[unused] = 1
                    unused += 1
            else:
                if nmask[i-1][j] != 0 and nmask[i][j-1] != 0:
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

                elif nmask[i-1][j] != 0:
                    new_mask[i][j] = new_mask[i-1][j]
                    areas[new_mask[i][j]] += 1
                elif nmask[i][j-1] != 0:
                    new_mask[i][j] = new_mask[i][j-1]
                    areas[new_mask[i][j]] += 1
                else:
                    new_mask[i][j] = unused
                    areas[unused] = 1
                    unused += 1
    return resolve, areas, new_mask


def solve(video, threshold):
    backSub = cv2.createBackgroundSubtractorMOG2()
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        fgmask = backSub.apply(frame) ##find foreground
        shadowval = backSub.getShadowValue() ## elminiate shadow
        ret, nmask = cv2.threshold(fgmask, shadowval, 255, cv2.THRESH_BINARY)
        if not ret:
            break
        
        # cv2.imshow("frame", nmask)
        # if cv2.waitKey(33) & 0xFF == ord('q'):
        #     break

        resolve, areas, cc_mask = connected_component(nmask)
        # np.set_printoptions(threshold=np.inf)
        # print(cc_mask)
        selected_frame = select_object(cc_mask, threshold, resolve, areas, cc_mask)

        cv2.imshow("frame", selected_frame)

        # cv2.circle(frame, (50, 50), 10, (255,0,0), 5)
        # cv2.circle(frame, (200, 200), 10, (0,255,0), 5)
        # cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':

    video = cv2.VideoCapture('input/train.mp4')
    if not video.isOpened():
        print('error : The video is not found.')
        exit(1)
        
    # minutes = 0
    # seconds = 30
    # frame_to_capture = int(minutes * 60 * video.get(cv2.CAP_PROP_FPS) + seconds * video.get(cv2.CAP_PROP_FPS))

    # # 设置帧号
    # video.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture)
    # ret, frame = video.read()
    # cv2.imwrite('input/3.jpg', frame)

    solve(video, threshold=1000)
    # test = np.array([[0,0,0,0,0,255,255,255,0,0],
    #                  [255,255,255,0,0,255,255,0,0,0],
    #                  [255,0,0,255,255,0,0,0,255,255],
    #                  [255,0,0,255,255,0,0,0,255,255]])
    # resolve, areas, cc_mask = connected_component(test)
    # print(resolve)
    # print('++++++++++++++++++')
    # print(areas)
    # print('++++++++++++++++++')
    # print(cc_mask)
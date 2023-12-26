import numpy as np
import cv2

def get_horizontal():
    return [
    [[1, 1, 1],
     [0, 0, 0],
     [0, 0, 0]],

    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]],

    [[0, 0, 0],
     [0, 0, 0],
     [1, 1, 1]]
    ]


def get_vertical():
    return [
    [[1, 0, 0],
     [1, 0, 0],
     [1, 0, 0]],

    [[0, 1, 0],
     [0, 1, 0],
     [0, 1, 0]],

    [[0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]]
    ]


def get_corner_ul():
    return [
    [[0, 1, 0],
     [1, 1, 0], 
     [0, 0, 0]],

    [[1, 0, 0],
     [0, 0, 0], 
     [0, 0, 0]],

    [[1, 1, 0], 
     [0, 0, 0], 
     [0, 0, 0]],

    [[1, 0, 0], 
     [1, 0, 0], 
     [0, 0, 0]],

    [[0, 0, 1],
     [1, 1, 1],
     [0, 0, 0]],

    # [[0, 1, 0], 
    #  [0, 1, 0], 
    #  [1, 1, 0]],

    # [[0, 0, 1], 
    #  [0, 0, 1], 
    #  [1, 1, 1]]
    ]

def get_corner_ur():
    return [
    [[0, 1, 0],
     [0, 1, 1], 
     [0, 0, 0]],

    [[1, 0, 0],
     [1, 0, 0], 
     [1, 1, 1]],

    [[0, 1, 0],
     [0, 1, 0], 
     [0, 1, 1]],

    [[1, 0, 0],
     [1, 1, 1], 
     [0, 0, 0]],

    [[0, 0, 1],
     [0, 0, 1], 
     [0, 0, 0]],

    [[0, 1, 1],
     [0, 0, 0], 
     [0, 0, 0]],

    [[0, 0, 1],
     [0, 0, 0], 
     [0, 0, 0]]
    ]


def get_corner_dl():
    return [
    [[0, 0, 0],
     [1, 1, 0],
     [0, 1, 0]],

    # [[1, 1, 0],
    #  [0, 1, 0], 
    #  [0, 1, 0]],

    # [[1, 1, 1],
    #  [0, 0, 1], 
    #  [0, 0, 1]],

    [[0, 0, 0],
     [1, 0, 0], 
     [1, 0, 0]],

    [[0, 0, 0],
     [1, 1, 1], 
     [0, 0, 1]],

    [[0, 0, 0],
     [0, 0, 0], 
     [1, 0, 0]],

    [[0, 0, 0],
     [0, 0, 0], 
     [1, 1, 0]]
    ]

def get_corner_dr():
    return [
    [[0, 0, 0],
     [0, 1, 1], 
     [0, 1, 0]],

    [[1, 1, 1],
     [1, 0, 0], 
     [1, 0, 0]],

    [[0, 1, 1],
     [0, 1, 0], 
     [0, 1, 0]],

    [[0, 0, 0],
     [1, 1, 1], 
     [1, 0, 0]],

    [[0, 0, 0],
     [0, 0, 1], 
     [0, 0, 1]],

    [[0, 0, 0],
     [0, 0, 0], 
     [0, 1, 1]],

    [[0, 0, 0],
     [0, 0, 0], 
     [0, 0, 1]]
    ]

def get_corner_ulr():
    return [
    [[0, 1, 0],
     [1, 1, 1], 
     [0, 0, 0]],

    [[1, 0, 0],
     [1, 0, 0], 
     [1, 1, 1]],

    [[0, 1, 0],
     [0, 1, 0], 
     [1, 1, 1]],

    [[0, 0, 1],
     [0, 0, 1], 
     [1, 1, 1]],

    [[1, 0, 0],
     [1, 1, 1], 
     [0, 0, 0]],

    [[0, 0, 1],
     [1, 1, 1], 
     [0, 0, 0]]
    ]

def get_corner_cana1():
    return [
    [[0, 0, 1],
     [1, 1, 1],
     [1, 0, 0]],

    [[0, 1, 0],
     [1, 1, 1],
     [1, 0, 0]]
    ]

def get_corner_cana2():
    return [
    [[1, 0, 0],
     [1, 1, 1],
     [0, 0, 1]]
    ]

def line_follower(frame, h, w, threshold=0.1):
    num_rows = 3
    num_cols = 3
    cell_height = h // num_rows
    cell_width = w // num_cols
    black = np.zeros((num_cols, num_rows))

    for i in range(num_rows):
        for j in range(num_cols):
            cell = frame[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

            black_pixels = np.sum(cell == 0)
            pixels = cell_height * cell_width

            if(black_pixels > pixels * threshold):
                black[i][j] = 1

            cv2.rectangle(frame, (j * cell_width, i * cell_height), ((j + 1) * cell_width, (i + 1) * cell_height), 0, 2)

    print(black)
    return black.tolist()
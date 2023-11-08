import keyboard
import cv2
import numpy as np
import time
# import tello
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

def main():
    drone = Tello()
    drone.connect()
    
    fs = cv2.FileSto
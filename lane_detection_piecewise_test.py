import cv2
import numpy as np
import carla
import torch

from lane_detection import LaneDetection

img = cv2.imread('test.png')
LD = LaneDetection()
LD.front2bev(img)


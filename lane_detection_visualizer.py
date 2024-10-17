from lane_detection import LaneDetection
from rgb_helper import Camera
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
import threading
import time
import cv2

frame = '0265910'

LD_module = LaneDetection()

fig = plt.figure("gt")
fig_test = plt.figure("test")
plt.ion()
plt.show()

image = cv2.imread("../data_imu/" + frame + ".jpg")
s = np.asarray(image)

splines = LD_module.lane_detection(s, fig_test)
LD_module.plot_state_lane(s, 0, fig)

def edge_detection(gray_image):
    gray_image = np.dot(gray_image, [0.333, 0.333, 0.333])

    gradient_sum = np.empty([gray_image.shape[0],gray_image.shape[1]])
    for i in range(gray_image.shape[0]):
        gradient_sum[i,:] = np.convolve([-1,0,1], gray_image[i,:], 'same')
    gradient_sum = np.absolute(gradient_sum)
    gradient_sum[:,0] = 0 # sees the start of the image as an edge, get rid of that
    gradient_sum[:,-1] = 0
    return np.sum(gradient_sum)

print(edge_detection(s)) # 500,000
image = cv2.imread("../data_imu/265440.jpg")
s = np.asarray(image)
print(edge_detection(s)) # 5,000

while(1):
    hello = 1

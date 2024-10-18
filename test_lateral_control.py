from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
from rgb_helper import Camera
import threading
import time
import cv2

# action variables

# init carla environement

# define variables
steps = 0

# init carla environement
camera = Camera()

# init modules of the pipeline
LD_module = LaneDetection()
LatC_module = LateralController()

# init extra plot
fig = plt.figure("gt")
fig_test = plt.figure("test")
plt.ion()
plt.show()

def start_camera(): # take input from a carla autopilot
    camera.run()

t1 = threading.Thread(target=start_camera, args=())

t1.start()
time.sleep(5)

index = 0
while True:
    # perform step

    if(index == camera.get_index()):
        continue
    index = camera.get_index()
    image = cv2.imread("frame.jpg")
    s = np.asarray(image)
    if(s.size != 320*240*3):
        continue
    f = open("speed.txt", "r")
    speed = float(f.read())

    # lane detection
    lane1, lane2 = LD_module.lane_detection(s, fig_test)

    # waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints)

    # control with constant gas and no braking
    a = LatC_module.stanley(waypoints, speed)

    #print("waypoints:")
    #print(waypoints)

    # output and plot figure
    if steps % 2 == 0:
        print("\naction " + str("{:+0.2f}".format(a)))
        print("targetspeed {:+0.2f}".format(target_speed))
        LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
    steps += 1

    # check if stop

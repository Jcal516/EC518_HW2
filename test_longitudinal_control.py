from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
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
camera = Camera()

# define variables
steps = 0

# init modules of the pipeline
LD_module = LaneDetection()
LatC_module = LateralController()
LongC_module = LongitudinalController()

# init extra plot
fig = plt.figure("gt")
fig_test = plt.figure("test")
plt.ion()
plt.show()

a = [0,0,0]

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
    lane1, lane2 = LD_module.lane_detection(s)#, fig_test)

    # waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints, max_speed=60, exp_constant=4.5)

    # control
    a[0] = LatC_module.stanley(waypoints, speed)
    a[1], a[2] = LongC_module.control(speed, target_speed)

    # output and plot figure
    if steps % 2 == 0:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("speed {:+0.2f} targetspeed {:+0.2f}".format(speed, target_speed))

        #LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
        LongC_module.plot_speed(speed, target_speed, steps, fig)
    steps += 1

    # check if stop

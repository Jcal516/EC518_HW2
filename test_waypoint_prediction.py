from lane_detection import LaneDetection
from rgb_helper import Camera
from waypoint_prediction import waypoint_prediction, target_speed_prediction
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
import threading
import time
import cv2

steps = 0

# init carla environement
camera = Camera()

# init modules of the pipeline
LD_module = LaneDetection()

# init extra plot
fig = plt.figure("gt")
fig_test = plt.figure("test")
plt.ion()
plt.show()

def start_camera():
    camera.run()

t1 = threading.Thread(target=start_camera, args=())

t1.start()
time.sleep(5)

index=0;
while True:
    # perform step
    
    # lane detection
    if(index == camera.get_index()):
        continue
    index = camera.get_index()
    image = cv2.imread("frame.jpg")
    s = np.asarray(image)
    if(not s.size):
        continue
    
    lane1, lane2 = LD_module.lane_detection(s, fig_test)

    # waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints)
    print(target_speed)
    # plot figure
    if steps % 2 == 0:
        LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
    steps += 1
    # check if stop

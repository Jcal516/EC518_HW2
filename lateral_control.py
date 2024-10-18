import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import math


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=.5, damping_constant=0.6):

        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0


    def stanley(self, waypoints, speed): 
        # assumes closest waypoints earliest in array
        # [0,:] = x coordinate
        # [1,:] = y coordinate
        '''
        ##### TODO #####
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            speed (float)
        '''
        # derive orientation error as the angle of the first path segment to the car orientation

        #waypoints = waypoints.T

        way_vec = waypoints[1,:] - waypoints[0,:]
        # TOA: tan(theta) = opposite/adjacent, opposite = x coord, adjacent = y coord from straight up along y axis
        o_e = math.atan(way_vec[0] / way_vec[1])

        # derive cross track error as distance between desired waypoint at spline parameter equal zero ot the car position

        ct_e = waypoints[0,0] - 320 / 2 # how far is the first waypoint from the center of the screen along the x axis?

        # derive stanley control law
        # prevent division by zero by adding as small epsilon
        '''print("o_e")
        print(o_e)
        print("atan")
        print(math.atan(self.gain_constant * ct_e / (speed + .000001)))'''

        control = o_e + math.atan(self.gain_constant * ct_e / (speed + .000001))

        # derive damping term
        
        steering_angle = control - self.damping_constant * (control - self.previous_steering_angle)
        self.previous_steering_angle = steering_angle

        # clip to the maximum stering angle (0.4) and rescale the steering action space
        return np.clip(steering_angle, -0.4, 0.4) / 0.4

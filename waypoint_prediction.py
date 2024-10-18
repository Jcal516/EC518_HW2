import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev, BSpline
from scipy.optimize import minimize
import time


def normalize(v):
    #print(v.shape)
    norm = np.linalg.norm(v,axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])

def curvature(waypoints):
    '''
    ##### TODO #####
    Curvature as the sum of the normalized dot product between the way elements
    Implement second term of the smoothin objective.

    args: 
        waypoints [2, num_waypoints] !!!!!
    '''
    b = 1
    curvature = 0
    for a in range(waypoints.shape[1]-2):
        vec1 = waypoints[:,a+1]-waypoints[:,a]

        vec2 = waypoints[:,a+2]-waypoints[:,a+1]
        #print(vec1)
        #print(vec2)
        v = np.append(vec1,vec2)
        v = normalize(v.reshape(2,2))
        dotproduct = np.dot(v[:,0],v[:,1])

        curvature += dotproduct
            
    curvature = curvature * b
    
    return curvature


def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    '''
    Objective for path smoothing

    args:
        waypoints [2 * num_waypoints] !!!!!
        waypoints_center [2 * num_waypoints] !!!!!
        weight_curvature (default=40)
    '''
    # mean least square error between waypoint and way point center
    #print(waypoints_center.shape)
    #print(waypoints)
    waypoints = waypoints.reshape(2,-1)
    ls_tocenter = np.mean((waypoints_center - waypoints)**2)

    # derive curvature
    curv = curvature(waypoints.reshape(2,-1))

    return -1 * weight_curvature * curv + ls_tocenter


def waypoint_prediction(roadside1_spline, roadside2_spline, num_waypoints=6, way_type = "smooth"):
    '''
    ##### TODO #####
    Predict waypoint via two different methods:
    - center
    - smooth 

    args:
        roadside1_spline
        roadside2_spline
        num_waypoints (default=6)
        parameter_bound_waypoints (default=1)
        waytype (default="smoothed")
    '''
    if way_type == "center":
        ##### TODO #####
     
        # create spline arguments
        print(type(roadside1_spline))
        t1,c1,k1 = roadside1_spline
        t2,c2,k2 = roadside2_spline
    

        
        new1_spline = BSpline(t1, np.array(c1).T, k1)
        new2_spline = BSpline(t2, np.array(c2).T, k2)
        # derive roadside points from spline
        x = np.linspace(0,1,num_waypoints)
        roadside1_points = np.array(splev(x,roadside1_spline)).T
        roadside2_points = np.array(splev(x,roadside2_spline)).T
        print(roadside1_points)
        print(roadside2_points)
        # derive center between corresponding roadside points
        way_points = (roadside1_points + roadside2_points) / 2
        print("way")
        print(way_points)
        print(way_points.shape)
        # output way_points with shape(2 x Num_waypoints)
        return way_points
    
    elif way_type == "smooth":
        ##### TODO #####

        # create spline arguments
        #print("smooth")
        #print(roadside1_spline)
        print(type(roadside1_spline))
        if isinstance(roadside1_spline, int):
            return roadside1_spline
        (t1,c1,k1) = roadside1_spline
        (t2,c2,k2) = roadside2_spline

        new1_spline = BSpline(t1, np.array(c1).T, k1)
        new2_spline = BSpline(t2, np.array(c2).T, k2)
        # derive roadside points from spline
        x = np.linspace(0,1,2*num_waypoints)
        roadside1_points = np.array(splev(x,roadside1_spline)).T
        roadside2_points = np.array(splev(x,roadside2_spline)).T
        #print(roadside1_points)
        #print(roadside2_points)
        # derive center between corresponding roadside points
        print(roadside1_points.shape)
        way_points_center = (roadside1_points.reshape(2,-1) + roadside2_points.reshape(2,-1)) / 2
        #print(way_points_center)
        # optimization
        way_points = minimize(smoothing_objective, 
                      (way_points_center), 
                      args=way_points_center)["x"]

        return way_points.reshape(-1,2)


def target_speed_prediction(waypoints, num_waypoints_used=5,
                            max_speed=60, exp_constant=.4, offset_speed=30):
    '''
    ##### TODO #####
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)
    
    output:
        target_speed (float)
    '''
    waypoints = waypoints.reshape(2,-1)
    curv = curvature(waypoints[:,:2*num_waypoints_used:2])
    print("curv=")
    print(curv)
    curv_new = num_waypoints_used-2-curv
    print("num_waypoints - 2 - curve = ")
    print(curv_new)
    print("exp = ")
    exponent = -1 * exp_constant * np.abs(curv)
    print(exponent)
    target_speed = ((max_speed - offset_speed) * np.exp(exponent)) + offset_speed

    return target_speed

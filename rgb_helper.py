import glob
import os
import sys
import time
from datetime import datetime, timedelta
import csv
import carla
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import logging
import random
from carla import ColorConverter as cc

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

class Camera:
    def __init__(self):
        self.index = 0

    def run(self):
        argparser = argparse.ArgumentParser(
            description=__doc__)
        argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        args = argparser.parse_args()

        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        try:
            world = client.get_world()
            ego_vehicle = None
            ego_cam = None
            ego_col = None
            ego_lane = None
            ego_obs = None
            ego_gnss = None
            ego_imu = None

            # --------------
            # Start recording
            # --------------
            client.start_recorder('C:/Users/jason/recording01.log')
            
            # --------------
            # Spawn ego vehicle
            # --------------
            ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
            ego_bp.set_attribute('role_name','ego')
            print('\nEgo role_name is set')
            ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
            ego_bp.set_attribute('color',ego_color)
            print('\nEgo color is set')

            spawn_points = world.get_map().get_spawn_points()
            number_of_spawn_points = len(spawn_points)

            if 0 < number_of_spawn_points:
                random.shuffle(spawn_points)
                ego_transform = spawn_points[0]
                ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
                print('\nEgo is spawned')
            else: 
                logging.warning('Could not found any spawn points')

            #--------------
            # Write to the csv file if it does not exist yet
            #--------------
            '''header = ['FileNum', 'Throttle', 'Steer','Brake']
            csv_file = 'C:/BU/EC518/Carla/CarlaSavedData/control.csv'
            if not os.path.exists(csv_file):
                with open('C:/BU/EC518/Carla/CarlaSavedData/control.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)'''
            
            # --------------
            # Add a RGB camera sensor to ego vehicle. 
            # --------------
            def print_img(image):
                if(image.frame % 10 == 0):
                    image.save_to_disk('frame.jpg')
                    self.index += 1

            cam_bp = None
            cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            cam_bp.set_attribute("image_size_x",str(320))
            cam_bp.set_attribute("image_size_y",str(240))
            cam_bp.set_attribute("fov",str(105))
            cam_bp.set_attribute("sensor_tick",str(100000))
            cam_location = carla.Location(2,0,1)
            cam_rotation = carla.Rotation(0,0,0)
            cam_transform = carla.Transform(cam_location,cam_rotation)
            ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
            ego_cam.listen(lambda image: print_img(image))

            # --------------
            # Add LiDAR sensor to ego vehicle
            # --------------
            '''def print_pc(pointc):
                """
                save the point cloud to disk every 10 frames
                pointc:     carla.LidarMeasurement
                """
                if(pointc.frame % 10 == 0):
                    pointc.save_to_disk('C:/BU/EC518/Carla/CarlaSavedData/%.6d.ply' % pointc.frame)

            cam_ld = None
            cam_ld = world.get_blueprint_library().find('sensor.lidar.ray_cast')
            ld_location = carla.Location(1,0,1)
            ld_rotation = carla.Rotation(0,0,0)
            ld_transform = carla.Transform(ld_location,ld_rotation)
            ego_ld = world.spawn_actor(cam_ld,ld_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
            ego_ld.listen(lambda pointc: print_pc(pointc))'''

            # --------------
            # Place spectator on ego spawning
            # --------------
            spectator = world.get_spectator()
            world_snapshot = world.wait_for_tick() 
            spectator.set_transform(ego_vehicle.get_transform())
            
            # --------------
            # Enable autopilot for ego vehicle
            # --------------
            ego_vehicle.set_autopilot(True)
            
            # --------------
            # Game loop. Prevents the script from finishing.
            # --------------
            while True:
                world_snapshot = world.wait_for_tick()

        finally:
            # --------------
            # Stop recording and destroy actors
            # --------------
            client.stop_recorder()
            if ego_vehicle is not None:
                if ego_cam is not None:
                    ego_cam.stop()
                    ego_cam.destroy()
                if ego_col is not None:
                    ego_col.stop()
                    ego_col.destroy()
                if ego_lane is not None:
                    ego_lane.stop()
                    ego_lane.destroy()
                if ego_obs is not None:
                    ego_obs.stop()
                    ego_obs.destroy()
                if ego_gnss is not None:
                    ego_gnss.stop()
                    ego_gnss.destroy()
                if ego_imu is not None:
                    ego_imu.stop()
                    ego_imu.destroy()
                #if ego_ld is not None:
                #    ego_ld.stop()
                #    ego_ld.destroy()
                ego_vehicle.destroy()

    '''if __name__ == '__main__':

        try:
            main()
        except KeyboardInterrupt:
            pass
        finally:
            print('\nDone with tutorial_ego.')'''
    
    def get_index(self):
        return self.index

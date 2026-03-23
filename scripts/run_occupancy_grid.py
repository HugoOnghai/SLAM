from slam_project.mapping import mapper as mp
from slam_project.localization import odometry as od
from slam_project.models.occupancy_grid import occupancy_grid
from slam_project.io import load_data as ld
import matplotlib.pyplot as plt
from slam_project.visualization import animations as anim

WORLD_BOUNDS = (-10, 10)  # meters
SCALE = 0.1  # meters per box
SCAN_IDX = 0

X, Y, theta, t = od.parse_encoders('./data/train/Encoders20')
lidar = ld.get_lidar('./data/train/Hokuyo20')

anim.animate_occupancy_grid(lidar[0:1], X, Y, theta, t)
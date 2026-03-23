from slam_project.io import load_data as ld
from slam_project.mapping import mapper as mp
import matplotlib.pyplot as plt
from slam_project.visualization import animations as anim
from slam_project.localization import odometry as od

lidar = ld.get_lidar('./data/train/Hokuyo20')

X, Y, theta, timestamps = od.parse_encoders_IMU('./data/train/Encoders20', './data/train/imu20')

fig, ax = plt.subplots(1,1,figsize=(10,10))

anim.global_lidar_timelapse(fig, ax, lidar, X, Y, theta, timestamps)
anim.topdown_trajectory(ax, X, Y)
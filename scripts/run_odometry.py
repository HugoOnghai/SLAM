from slam_project.io import load_data as ld
from slam_project.localization import odometry as od
from slam_project import config
import numpy as np
from slam_project.visualization import animations as anim
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(10,10))

encodermats = [
    './data/Encoders20',
    './data/Encoders21',
    './data/Encoders23'
]

imumats = [
    './data/imu20',
    './data/imu21',
    './data/imu23'
]

for (path_enc, path_imu) in zip(encodermats, imumats):
    # load encoder data
    X, Y, theta, ts = od.parse_encoders_IMU(path_enc, path_imu)

    anim.topdown_trajectory(ax, X, Y)

ax.legend()
plt.show()

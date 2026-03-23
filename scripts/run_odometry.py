from slam_project.io import load_data as ld
from slam_project.localization import odometry as od
from slam_project import config
import numpy as np
from slam_project.visualization import animations as anim
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(10,10))

encodermats = [
    './data/train/Encoders20',
    # './data/train/Encoders21',
    # './data/train/Encoders23'
]

for path in encodermats:
    # load encoder data
    X, Y, theta, ts = od.parse_encoders(path)

    anim.topdown_trajectory(ax, X, Y)

ax.legend()
plt.show()

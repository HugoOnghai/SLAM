from slam_project.io import load_data as ld
from slam_project.mapping import mapper as mp
import matplotlib.pyplot as plt
from slam_project.visualization import animations as anim
from slam_project.localization import odometry as od

room_name = "23"

lidar = ld.get_lidar(f'./data/Hokuyo{room_name}')

X, Y, theta, timestamps = od.parse_encoders_IMU(f'./data/Encoders{room_name}', f'./data/imu{room_name}')

fig, ax = plt.subplots(1,1,figsize=(10,10))

anim.global_lidar_timelapse(fig, ax, lidar, X, Y, theta, timestamps)
ax.plot(X[0], Y[0], color='green', label="Start", marker='o', markersize=10)
ax.plot(X[-1], Y[-1], color='red', label="End", marker='x', markersize=10)
ax.set_title(f"Robot's Motion/Path in Train{room_name}, from Dead Reckoning")
ax.legend()
fig.tight_layout()
ax.set_xlabel("Displacement along the x-axis (m)")
ax.set_ylabel("Displacement along the y-axis (m)")

fig.savefig(f"./outputs/figures/lidar+deadreckoning_nonSLAM_Train{room_name}.png")
from slam_project.mapping import mapper as mp
from slam_project.localization import odometry as od
from slam_project.models.occupancy_grid import occupancy_grid
from slam_project.io import load_data as ld
import matplotlib.pyplot as plt
from slam_project.visualization import animations as anim

WORLD_BOUNDS = (-10, 30)  # meters
SCALE = 0.1  # meters per box
SCAN_IDX = 0

room_name = "23"
X, Y, theta, t = od.parse_encoders_IMU(f'./data/train/Encoders{room_name}', f'./data/train/imu{room_name}')
lidar = ld.get_lidar(f'./data/train/Hokuyo{room_name}')


# grid = occupancy_grid(0, 0, SCALE, WORLD_BOUNDS)
# i = mp.nearest_pose_index(lidar[0]['t'], t)
# ox, oy = mp.lidar_hits_global(lidar[0], X[i], Y[i], theta[i])
# grid.add_scan_hits(ox, oy, X[i], Y[i], runBresenham=True)
# grid.plot()

fig, ax = plt.subplots(figsize=(8, 8))

my_scale = 0.1
anim.animate_occupancy_grid(fig, ax, lidar, X, Y, theta, t, scale=my_scale, world_bounds=WORLD_BOUNDS)
ax.set_title(f"Occupancy Grid for {room_name}")
fig.tight_layout()

file_name = f"./outputs/figures/occupancygrid_nonSLAM_Train{room_name}.png"
fig.savefig(file_name)
print(f"Output saved to: {file_name}")
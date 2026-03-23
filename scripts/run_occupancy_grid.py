from slam_project.mapping import mapper as mp
from slam_project.localization import odometry as od
from slam_project.models.occupancy_grid import occupancy_grid
from slam_project.io import load_data as ld
import matplotlib.pyplot as plt
from slam_project.visualization import animations as anim

WORLD_BOUNDS = (-10, 25)  # meters
SCALE = 0.1  # meters per box
SCAN_IDX = 0

X, Y, theta, t = od.parse_encoders_IMU('./data/train/Encoders20', './data/train/imu20')
lidar = ld.get_lidar('./data/train/Hokuyo20')

# grid = occupancy_grid(0, 0, SCALE, WORLD_BOUNDS)
# i = mp.nearest_pose_index(lidar[0]['t'], t)
# ox, oy = mp.lidar_hits_global(lidar[0], X[i], Y[i], theta[i])
# grid.add_scan_hits(ox, oy, X[i], Y[i], runBresenham=True)
# grid.plot()

fig, ax = plt.subplots(figsize=(8, 8))

anim.animate_occupancy_grid(fig, ax, lidar, X, Y, theta, t, scale=0.1, world_bounds=WORLD_BOUNDS)

fig.savefig("./outputs/figures/occupancygrid_nonSLAM.png")
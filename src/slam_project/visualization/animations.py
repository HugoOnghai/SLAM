import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from slam_project.mapping import mapper as mp
from slam_project.models.occupancy_grid import occupancy_grid

# ODOMETRY
def topdown_trajectory(ax, x_data, y_data):
    ax.plot(x_data, y_data)
    ax.plot(x_data[0], y_data[0], color='green', label="Start", marker='o', markersize=20)
    ax.plot(x_data[-1], y_data[-1], color='red', label="End", marker='o', markersize=20)
    # ax.axis("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Robot Trajectory")
    ax.grid(True)
    plt.draw()

# MAPPING
def lidar_timelapse(lidar, stride=10):
    fig, ax = plt.subplots(1,1,figsize=(10,10))

    for i in range(200, len(lidar), 10):
        mp.lidar2gridmap(ax, lidar[i], stride=stride)
        plt.draw()
        plt.pause(0.001)
        ax.clear()

def global_lidar_timelapse(fig, ax, lidar, X, Y, theta, theta_timestamps):
    '''
    lidar is the lidar scans over time
    theta is the integrated global rotation over time
    combine both so "x-axis" is in inertial frame not body frame
    '''

    X_max = np.max(X)
    X_min = np.min(X)
    Y_max = np.max(Y)
    Y_min = np.min(Y)

    dim_max = np.max([X_max, Y_max]) + 1
    dim_min = np.min([X_min, Y_min]) - 1

    for i in range(200, len(lidar), 10):
        ax.clear()
        ax.set_xlim(dim_min, dim_max)
        ax.set_ylim(dim_min, dim_max)
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale(False)

        lidar_snapshot = lidar[i]

        # since encoders and lidar aren't synchronized, just find nearest timestamps (might change later)
        latency = np.abs(theta_timestamps - lidar[i]['t'])
        nearest_time_index = np.argmin(latency)
        theta_snapshot = theta[nearest_time_index]
        X_snapshot = X[nearest_time_index]
        Y_snapshot = Y[nearest_time_index]

        ax.plot(X[0:nearest_time_index], Y[0:nearest_time_index], c='green', linestyle='-')
        mp.global_lidar2gridmap(ax, lidar_snapshot, X_snapshot, Y_snapshot, theta_snapshot, stride=10)
        plt.draw()
        plt.pause(0.005)

def animate_occupancy_grid(fig, ax, lidar, X, Y, theta, t, scale=0.1, world_bounds=(-20, 20)):
    grid = occupancy_grid(0, 0, scale, world_bounds)
    
    image = ax.imshow(grid.grid, origin="lower", vmin=0, vmax=1)
    plt.colorbar(image, ax=ax)

    for i in range(0, len(lidar), 10):
        lidar_snapshot = lidar[i]

        nearest_idx = mp.nearest_pose_index(lidar_snapshot["t"], t)
        ox, oy = mp.lidar_hits_global(
            lidar_snapshot,
            X[nearest_idx],
            Y[nearest_idx],
            theta[nearest_idx],
        )

        i = mp.nearest_pose_index(lidar_snapshot['t'], t)

        ox_grid, oy_grid, rx_grid, ry_grid = grid.add_scan_hits(ox, oy, X[i], Y[i], runBresenham=True)

        image.set_data(grid.grid)
        ax.plot()
        ax.set_title(f"Scan {i}: Current Robot Position ({X[i]:.2f}, {Y[i]:.2f})")
        plt.draw()
        plt.pause(0.01)

    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import slam_project.config as config


def nearest_pose_index(scan_time, odometry_timestamps):
	# "synchronize" the scan_time and odometry_timestamps
	latency = np.abs(odometry_timestamps - scan_time)
	return np.argmin(latency)


def lidar_hits_global(lidar_snapshot, x, y, theta, res=10):
	# converts angles+distances to global xy-coords
	angles = np.asarray(lidar_snapshot['angle'][::res]).ravel()
	distances = np.asarray(lidar_snapshot['scan'][::res]).ravel() # in meters!

	valid_hits = np.isfinite(distances) & (distances >= config.LIDAR_MIN_RANGE) & (distances <= config.LIDAR_MAX_RANGE)
	angles = angles[valid_hits]
	distances = distances[valid_hits]

	ox = np.cos(angles + theta) * distances + (x) # assume that robot position in meters
	oy = np.sin(angles + theta) * distances + (y) # since it was converted in od.parse_encoders

	return ox, oy


def lidar2gridmap(ax, lidar_snapshot, stride=10):
	# stride is the resolution of your map, larger stride is less dots
	angles = np.asarray(lidar_snapshot['angle']).ravel()
	distances = np.asarray(lidar_snapshot['scan']).ravel()
	distances[distances > 30] = np.nan

	ox = np.cos(angles) * distances
	oy = np.sin(angles) * distances

	ax.scatter(ox[::stride], oy[::stride], c="r", s=5)
	ax.axis("equal")
	bottom, top = plt.ylim()  # return the current ylim
	# ax.set_ylim((top, bottom)) # rescale y axis, to match the grid orientation
	ax.set_xlim(-30, 30)
	ax.set_ylim(-30, 30)
	ax.grid(True)

def global_lidar2gridmap(ax, lidar_snapshot, X_snapshot, Y_snapshot, theta_snapshot, stride=10, returnCoords=False):
	'''
	assuming initial theta of 0, rotate lidar snapshot by theta snapshot to convert
	lidar scan grid map from robot body frame x-y to global/inertial x-y
	'''
	ox, oy = lidar_hits_global(lidar_snapshot, X_snapshot, Y_snapshot, theta_snapshot)

	if returnCoords:
		return ox, oy
	
	ax.scatter(ox[::stride], oy[::stride], c="r", s=5)
	# bottom, top = plt.ylim()  # return the current ylim
	# ax.set_ylim((top, bottom)) # rescale y axis, to match the grid orientation
	ax.grid(True)
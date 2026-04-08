from slam_project.localization import odometry as od

enc_path   = './data/Encoders20'
imu_path   = './data/imu20'
lidar_path = './data/Hokuyo20'

od.parse_deltas_at_lidar_times(enc_path, imu_path, lidar_path)
from slam_project.localization import odometry as od

enc_path   = './data/train/Encoders20'
imu_path   = './data/train/imu20'
lidar_path = './data/train/Hokuyo20'

od.parse_deltas_at_lidar_times(enc_path, imu_path, lidar_path)
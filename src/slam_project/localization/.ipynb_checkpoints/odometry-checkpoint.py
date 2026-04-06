import numpy as np
from slam_project.localization import odometry as od
from slam_project import config
from slam_project.io import load_data as ld

def integrate(initial, delta):
    net = np.cumsum(delta)
    absolute = net + initial # should add initial to every element in net
    return absolute

def wheel2deltaOXY(s_L, s_R, angle, deltaAngle, enc_ts):
    if len(s_L) != len(s_R):
        raise ValueError("encoder counts don't match in size!")
    elif len(s_L) != len(angle):
        raise ValueError("encoder count length doesn't match angle array length!")

    intermediate_angle = angle - (deltaAngle/2)

    deltaX = ((s_L+s_R)/2) * (np.cos(intermediate_angle))
    deltaY = ((s_L+s_R)/2) * (np.sin(intermediate_angle))

    return deltaX, deltaY

def parse_encoders_IMU(encoder_path, IMU_path):
    '''
    computes global X and Y and theta
    '''
    FL, FR, RL, RR, enc_ts = ld.get_encoder(encoder_path)
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_ts = ld.get_imu(IMU_path)

    FL = kalman_filter(FL)
    FR = kalman_filter(FR)
    RL = kalman_filter(RL)
    RR = kalman_filter(RR)

    acc_x = kalman_filter(acc_x)
    acc_y = kalman_filter(acc_y)
    acc_z = kalman_filter(acc_z)
    gyro_x = kalman_filter(gyro_x)
    gyro_y = kalman_filter(gyro_y)
    gyro_z = kalman_filter(gyro_z)
    
    # convert per-wheel encoder ticks to per-side encoder ticks
    # currently I just only consider the front wheels
    e_L = (FL + RL)/2
    e_R = (FR + RR)/2

    assert len(e_L) == len(e_R)

    # convert encoder ticks to arch lengths
    s_L = e_L * (1/360) * (np.pi * config.WHEEL_DIAMETER) # mm
    s_R = e_R * (1/360) * (np.pi * config.WHEEL_DIAMETER) # mm

    # width between wheels
    w = config.BODY_WIDTH # mm

    # calculate angular odometry
    deltaTheta_enc = (s_R-s_L)/(w) # unitless becomes radians turned at each given time step

    # integrate deltaTheta to get global rotation
    theta_enc = od.integrate(0, deltaTheta_enc)

    ### INCORPORATE IMU DATA
    dt_imu = np.diff(imu_ts, prepend=imu_ts[0])
    deltaTheta_imu = gyro_z * dt_imu
    theta_imu = integrate(0, deltaTheta_imu)
    theta_imu_on_enc = np.interp(enc_ts, imu_ts, theta_imu)

    # convert encoder ticks to deltaX and deltaY in each time interval
    # I saw in literature that people called the inertial frame of reference OXY
    # origin-x-y axes
    alpha = 0.50
    theta_fused = alpha * theta_imu_on_enc + (1 - alpha) * theta_enc
    deltaX, deltaY = od.wheel2deltaOXY(s_L, s_R, theta_fused, deltaTheta_enc, enc_ts)

    # convert position deltas to positions with integration 
    X = od.integrate(0, deltaX)
    Y = od.integrate(0, deltaY)

    # convert to meters
    X /= 1000
    Y /= 1000

    return X, Y, theta_fused, enc_ts

def compute_body_frame_odometry_at_lidar_times(encoder_path, imu_path, lidar_path):
    '''
    This function is the backbone of my particle filter odometry step. It computes...
    - the deltaX, deltaY, and deltaTheta between each timestep of the lidar scan.
    - we use the lidar scan timesteps as a baseline.
    '''
    ## GRAB RAW DATA
    FL, FR, RL, RR, enc_ts = ld.get_encoder(encoder_path)
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, imu_ts = ld.get_imu(imu_path)
    lidar = ld.get_lidar(lidar_path)

    enc_ts = np.asarray(enc_ts)
    imu_ts = np.asarray(imu_ts)
    lidar_ts = np.asarray([scan['t'] for scan in lidar])

    ## KALMAN FILTER
    FL = kalman_filter(FL)
    FR = kalman_filter(FR)
    RL = kalman_filter(RL)
    RR = kalman_filter(RR)

    acc_x = kalman_filter(acc_x)
    acc_y = kalman_filter(acc_y)
    acc_z = kalman_filter(acc_z)
    gyro_x = kalman_filter(gyro_x)
    gyro_y = kalman_filter(gyro_y)
    gyro_z = kalman_filter(gyro_z)

    ## ENCODER TICKS --> ARCH LENGTHS
    e_L = (FL + RL)/2
    e_R = (FR + RR)/2
    s_L = e_L * (1/360) * (np.pi * config.WHEEL_DIAMETER) # mm
    s_R = e_R * (1/360) * (np.pi * config.WHEEL_DIAMETER) # mm
    w = config.BODY_WIDTH # mm

    ## ROTATION FROM ENCODERS AND IMU
    delta_theta_enc = (s_R - s_L) / w # ENCODER DATA
    dt_imu = np.diff(imu_ts, prepend=imu_ts[0])
    delta_theta_imu = gyro_z * dt_imu # IMU DATA
    delta_theta_imu_on_enc = np.interp(enc_ts, imu_ts, delta_theta_imu) # interpolated onto enc_ts

    ## FUSE ROTATION DELTAS
    alpha = 0.50 # trust imu more than encoders since its skid-steer
    delta_theta_fused_enc = alpha * delta_theta_imu_on_enc + (1 - alpha) * delta_theta_enc

    ## Approximate Forward Movement (Y_BODY)
    delta_y_body_enc = 0.5 * (s_L + s_R)
    delta_x_body_enc = np.zeros_like(delta_y_body_enc) 
    ### I make the assumption that the robot won't move perpendicularly between time steps
    ### I hope LIDAR scans fast enough where forward movement and heading direction change is sufficient...

    ## ACCUMULATE TO INTERPOLATE ONTO LIDAR TIMES
    y_body_enc = integrate(0, delta_y_body_enc)
    x_body_enc = integrate(0, delta_x_body_enc) # should remain zero right now
    theta_enc = integrate(0, delta_theta_fused_enc) 

    y_body_lidar = np.interp(lidar_ts, enc_ts, y_body_enc)
    x_body_lidar = np.interp(lidar_ts, enc_ts, x_body_enc)
    theta_lidar = np.interp(lidar_ts, enc_ts, theta_enc)

    ## DIFFERENCE AT LIDAR TIMES in METERS
    delta_y_body_at_lidar = np.diff(y_body_lidar, prepend=y_body_lidar[0]) / 1000
    delta_x_body_at_lidar = np.diff(x_body_lidar, prepend=x_body_lidar[0]) / 1000
    delta_theta_at_lidar = np.diff(theta_lidar, prepend=theta_lidar[0]) / 1000

    return delta_y_body_at_lidar, delta_x_body_at_lidar, delta_theta_at_lidar, lidar_ts

# MY KALMAN FILTER FROM PROJECT 2
def kalman_filter(raw, Q=1e-5, R=1e-2):
    n = len(raw)
    x_hat = np.zeros(n) # what we predict the raw data "should" be if there was no measurement noise
    P = np.zeros(n) # the uncertainties of our predicted noise-free data

    # initialization
    x_hat[0] = raw[0] # assume that the first measurement is what it should be
    P[0] = 1.0 # and that we make this assumption with 100% uncertainty

    for k in range(1,n): # for every subsequent measurement
        # predict what the next measurement should be, based on the previous k-1
        x_pred = x_hat[k-1]
        P_pred = P[k-1] + Q # our uncertainty increases since we propagate our uncertainty from k-1 to measurement k

        # update our prediction and uncertainty from k-1 based on what the measurement k actually was
        K = P_pred / (P_pred + R) # kalman gain determines our smoothing, based on R which is a parameter set to reflect the noisiness of the measurement
        x_hat[k] = x_pred + K * (raw[k] - x_pred)
        P[k] = (1 - K) * P_pred

    return x_hat
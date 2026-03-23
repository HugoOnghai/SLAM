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

def parse_encoders(encoder_path):
    '''
    computes global X and Y and theta
    '''
    FL, FR, RL, RR, enc_ts = ld.get_encoder(encoder_path)

    FL = kalman_filter(FL)
    FR = kalman_filter(FR)
    RL = kalman_filter(RL)
    RR = kalman_filter(RR)
    enc_ts = kalman_filter(enc_ts)

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
    deltaTheta = (s_R-s_L)/(w) # unitless becomes radians turned at each given time step

    # integrate deltaTheta to get global rotation
    theta = od.integrate(0, deltaTheta)

    # convert encoder ticks to deltaX and deltaY in each time interval
    # I saw in literature that people called the inertial frame of reference OXY
    # origin-x-y axes
    deltaX, deltaY = od.wheel2deltaOXY(s_L, s_R, theta, deltaTheta, enc_ts)

    # convert position deltas to positions with integration 
    X = od.integrate(0, deltaX)
    Y = od.integrate(0, deltaY)

    # convert to meters
    X /= 1000
    Y /= 1000

    return X, Y, theta, enc_ts

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
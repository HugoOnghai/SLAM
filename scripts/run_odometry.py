from slam_project.io import load_data as ld
from slam_project.localization import odometry as od
from slam_project import config
import numpy as np
from slam_project.visualization import animations as anim
import matplotlib.pyplot as plt

# load encoder data
FL, FR, RL, RR, enc_ts = ld.get_encoder('./data/train/Encoders20')

# convert per-wheel encoder ticks to per-side encoder ticks
# currently I just only consider the front wheels
e_L = FL
e_R = FR

assert len(e_L) == len(e_R)

# width between wheels
w = config.BODY_WIDTH

# calculate angular odometry
deltaTheta = (e_R-e_L)/(w)

# integrate deltaTheta to get global rotation
theta = od.integrate(0, deltaTheta)

# convert encoder ticks to deltaX and deltaY in each time interval
# I saw in literature that people called the inertial frame of reference OXY
# origin-x-y axes
deltaX, deltaY = od.ticks2deltaOXY(e_L, e_R, deltaTheta, enc_ts)

# convert position deltas to positions with integration 
X = od.integrate(0, deltaX)
Y = od.integrate(0, deltaY)

fig, ax = plt.subplots(1,1)
anim.topdown_trajectory(ax, X, Y)
ax.legend()

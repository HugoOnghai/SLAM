import numpy as np

def integrate(initial, delta):
    net = np.cumsum(delta)
    absolute = net + initial # should add initial to every element in net
    return absolute

def ticks2deltaOXY(e_L, e_R, angle, enc_ts):
    if len(e_L) != len(e_R):
        raise ValueError("encoder counts don't match in size!")
    elif len(e_L) != len(angle):
        raise ValueError("encoder count length doesn't match angle array length!")

    deltaX = ((e_L+e_R)/2) * (np.cos(angle))
    deltaY = ((e_L+e_R)/2) * (np.sin(angle))

    return deltaX, deltaY
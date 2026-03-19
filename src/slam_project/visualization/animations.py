import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def topdown_trajectory(ax, x_data, y_data):
    ax.plot(x_data, y_data)
    ax.plot(x_data[0], y_data[0], label="Start")
    ax.plot(x_data[-1], y_data[-1], label="End")
    ax.axis("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Robot Trajectory")
    ax.grid(True)
    plt.show()

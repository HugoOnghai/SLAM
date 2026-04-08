import numpy as np
import matplotlib.pyplot as plt
import MapUtils_fclad as mu

class occupancy_grid:

    # def __init__(self, GRID_DIM):
    #     self.grid = np.ones([GRID_DIM, GRID_DIM]) * 0.5

    def __init__(self, origin_x, origin_y, scale, WORLD_DIM, name="Occupancy Grid"):
        self.name = name
        self.origin_x = origin_x # middle
        self.origin_y = origin_y # middle
        self.scale = scale # meters per box
        self.WORLD_DIM = WORLD_DIM # meters
        self.GRID_DIM = int(np.abs(self.WORLD_DIM[1] - self.WORLD_DIM[0])/self.scale) # boxes along axis
        self.grid = np.zeros([self.GRID_DIM, self.GRID_DIM])

    def init_live_plot(self, robot_x=None, robot_y=None):
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.grid, origin="lower")
        self.fig.colorbar(self.im, ax=self.ax)
        self.robot_x_array = []
        self.robot_y_array = []

        if robot_x is None or robot_y is None:
            self.robot_marker = self.ax.scatter([], [], c="red", s=80, marker="x")
        else:
            self.robot_marker = self.ax.scatter([robot_x], [robot_y], c="red", s=80, marker="x")
            self.robot_x_array.append(robot_x)
            self.robot_y_array.append(robot_y)

        self.ax.set_title(f"{self.name} with SLAM")

        # plt.show(block=False)

    # def update_live_plot(self, robot_x_grid: int, robot_y_grid: int):
    #     self.im.set_data(self.grid)
    #     self.robot_marker.set_offsets([[robot_x_grid, robot_y_grid]])
    #     self.ax.figure.canvas.draw_idle()
    #     plt.pause(0.001)
    
    def update_live_plot(self, robot_x_meters: float, robot_y_meters: float):
        origin_x_grid = int(
            (abs(self.origin_x - self.WORLD_DIM[0]) / abs(self.WORLD_DIM[1] - self.WORLD_DIM[0]))
            * self.GRID_DIM
        )
        origin_y_grid = int(
            (abs(self.origin_y - self.WORLD_DIM[0]) / abs(self.WORLD_DIM[1] - self.WORLD_DIM[0]))
            * self.GRID_DIM
        )

        # plot the occupancy grid with the robot on top of it
        robot_x_grid = np.floor((robot_x_meters - self.origin_x) / self.scale).astype(int) + origin_x_grid
        robot_y_grid = np.floor((robot_y_meters - self.origin_y) / self.scale).astype(int) + origin_y_grid

        self.robot_y_array.append(robot_y_grid)
        self.robot_x_array.append(robot_x_grid)

        self.im.set_data(self.grid)
        self.robot_marker.set_offsets([[robot_x_grid, robot_y_grid]])

        self.ax.figure.canvas.draw_idle()
        plt.pause(0.0001)

    def add_traj_live_plot(self, filename):
        self.ax.plot(self.robot_x_array, self.robot_y_array, color="green")
        plt.savefig(filename)
        # plt.pause(100)
        plt.show()

    def plot(self):
        # alternative to live map where u just call it easily
        plt.imshow(self.grid, origin="lower")
        plt.colorbar()
        plt.show()

    def plot(self, fig, ax, robot_x, robot_y):
        origin_x_grid = int(
            (abs(self.origin_x - self.WORLD_DIM[0]) / abs(self.WORLD_DIM[1] - self.WORLD_DIM[0]))
            * self.GRID_DIM
        )
        origin_y_grid = int(
            (abs(self.origin_y - self.WORLD_DIM[0]) / abs(self.WORLD_DIM[1] - self.WORLD_DIM[0]))
            * self.GRID_DIM
        )

        plt.imshow(self.grid, origin="lower")
        plt.colorbar()

        if robot_x is not None or robot_y is not None:
            # plot the occupancy grid with the robot on top of it
            robot_x_grid = np.floor((robot_x - self.origin_x) / self.scale).astype(int) + origin_x_grid
            robot_y_grid = np.floor((robot_y - self.origin_y) / self.scale).astype(int) + origin_y_grid

            plt.scatter(robot_x_grid, robot_y_grid, c="red", s=80, marker="x", label="robot")
        
        plt.legend()

    def hits_to_grid_cells(self, ox, oy):
        ox = np.asarray(ox)
        oy = np.asarray(oy)

        origin_x_grid = int(
            (abs(self.origin_x - self.WORLD_DIM[0]) / abs(self.WORLD_DIM[1] - self.WORLD_DIM[0]))
            * self.GRID_DIM
        )
        origin_y_grid = int(
            (abs(self.origin_y - self.WORLD_DIM[0]) / abs(self.WORLD_DIM[1] - self.WORLD_DIM[0]))
            * self.GRID_DIM
        )

        ox_grid = np.floor((ox - self.origin_x) / self.scale).astype(int) + origin_x_grid
        oy_grid = np.floor((oy - self.origin_y) / self.scale).astype(int) + origin_y_grid

        valid_hits = (
            (ox_grid >= 0) & (ox_grid < self.GRID_DIM) &
            (oy_grid >= 0) & (oy_grid < self.GRID_DIM)
        )

        return ox_grid[valid_hits], oy_grid[valid_hits]
    
    def miss_to_grid_cells(self, ox_grid, oy_grid, robot_x, robot_y):
        '''
        returns the free cells determined by Bresenham between lidar hits and robot.
        '''
        origin_x_grid = ((np.abs(self.origin_x - self.WORLD_DIM[0])/np.abs(self.WORLD_DIM[1]-self.WORLD_DIM[0]))*self.GRID_DIM).astype(int)
        origin_y_grid = ((np.abs(self.origin_y - self.WORLD_DIM[0])/np.abs(self.WORLD_DIM[1]-self.WORLD_DIM[0]))*self.GRID_DIM).astype(int)
        robot_x_grid = np.floor((robot_x - self.origin_x) / self.scale).astype(int) + origin_x_grid
        robot_y_grid = np.floor((robot_y - self.origin_y) / self.scale).astype(int) + origin_y_grid

        valid_hits = (
            (ox_grid >= 0) & (ox_grid < self.GRID_DIM) &
            (oy_grid >= 0) & (oy_grid < self.GRID_DIM)
        )

        if not np.any(valid_hits):
            empty = np.empty(0, dtype=np.intp)
            return empty, empty

        robot_in_bounds = (
            0 <= robot_x_grid < self.GRID_DIM and
            0 <= robot_y_grid < self.GRID_DIM
        )

        if ox_grid.size and robot_in_bounds:
            x_miss, y_miss = bresenham(robot_x_grid, robot_y_grid, ox_grid, oy_grid)
            valid_ray = (
                (x_miss >= 0) & (x_miss < self.GRID_DIM) &
                (y_miss >= 0) & (y_miss < self.GRID_DIM)
            )
            x_miss = x_miss[valid_ray]
            y_miss = y_miss[valid_ray]

            return x_miss, y_miss
        else:
            empty = np.empty(0, dtype=np.intp)
            return empty, empty

    
    def add_scan_hits(self, ox, oy, robot_x, robot_y, runBresenham=True):
        ox = np.asarray(ox)
        oy = np.asarray(oy)

        dx = ox - self.origin_x
        dy = oy - self.origin_y

        origin_x_grid = ((np.abs(self.origin_x - self.WORLD_DIM[0])/np.abs(self.WORLD_DIM[1]-self.WORLD_DIM[0]))*self.GRID_DIM).astype(int)
        origin_y_grid = ((np.abs(self.origin_y - self.WORLD_DIM[0])/np.abs(self.WORLD_DIM[1]-self.WORLD_DIM[0]))*self.GRID_DIM).astype(int)
        robot_x_grid = np.floor((robot_x - self.origin_x) / self.scale).astype(int) + origin_x_grid
        robot_y_grid = np.floor((robot_y - self.origin_y) / self.scale).astype(int) + origin_y_grid
        ox_grid = np.floor(dx / self.scale).astype(int) + origin_x_grid
        oy_grid = np.floor(dy / self.scale).astype(int) + origin_y_grid


        valid_hits = (
            (ox_grid >= 0) & (ox_grid < self.GRID_DIM) &
            (oy_grid >= 0) & (oy_grid < self.GRID_DIM)
        )

        if not np.any(valid_hits):
            empty = np.empty(0, dtype=np.intp)
            return empty, empty, robot_x_grid, robot_y_grid

        ox_grid = ox_grid[valid_hits]
        oy_grid = oy_grid[valid_hits]

        robot_in_bounds = (
            0 <= robot_x_grid < self.GRID_DIM and
            0 <= robot_y_grid < self.GRID_DIM
        )

        if runBresenham and ox_grid.size and robot_in_bounds:
            cols, rows = bresenham(robot_x_grid, robot_y_grid, ox_grid, oy_grid)
            valid_ray = (
                (cols >= 0) & (cols < self.GRID_DIM) &
                (rows >= 0) & (rows < self.GRID_DIM)
            )
            cols = cols[valid_ray]
            rows = rows[valid_ray]
            self.grid[rows, cols] -= 0.1 # line to wall should be free space

        self.grid[oy_grid, ox_grid] += 0.5

        self.grid = np.clip(self.grid, -5.0, 5.0)

        return ox_grid, oy_grid, robot_x_grid, robot_y_grid

def bresenham(x0, y0, x1, y1):
    """
    Given two grid cells, find the best straight line approximation btwn them
    source: https://github.com/encukou/bresenham/blob/master/bresenham.py
    Don't use this anymore, try using the maputils cython one now
    """

    xends = np.ascontiguousarray(np.atleast_1d(x1), dtype=np.int16)
    yends = np.ascontiguousarray(np.atleast_1d(y1), dtype=np.int16)

    if xends.shape != yends.shape:
        raise ValueError("x1 and y1 must have matching shapes")

    if xends.size == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty

    max_map = int(max(x0, y0, np.max(xends), np.max(yends))) + 1
    ray = mu.getMapCellsFromRay_fclad(
        int(x0),
        int(y0),
        xends,
        yends,
        max_map,
    )

    return ray[0].astype(np.intp, copy=False), ray[1].astype(np.intp, copy=False)


    # # assumes that x0,y0,x1,y1 are all grid points and therefore integers.

    # dx = x1 - x0
    # dy = y1 - y0

    # xsign = 1 if dx > 0 else -1
    # ysign = 1 if dy > 0 else -1

    # dx = np.abs(dx)
    # dy = np.abs(dy)

    # if dx > dy:
    #     xx, xy, yx, yy = xsign, 0, 0, ysign
    # else:
    #     dx, dy = dy, dx
    #     xx, xy, yx, yy = 0, ysign, xsign, 0

    # D = 2*dy - dx # error term move straight or diagonally?
    # y = 0

    # line_coords = []

    # for x in range(dx + 1):
    #     line_coords.append((x0 + x*xx + y*yx, y0 + x*xy + y*yy))
    #     if D >= 0: # increment y if error is larger along y than x
    #         y += 1
    #         D -= 2*dx
    #     D += 2*dy

    # return line_coords

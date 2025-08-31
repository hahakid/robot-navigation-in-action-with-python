"""
Normal Distribution Transform (NDTGrid) mapping sample
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
# from lesson3.grid_map_lib.grid_map_lib import GridMap
from utils.plot import plot_covariance_ellipse


class FloatGrid:

    def __init__(self, init_val=0.0):
        self.data = init_val

    def get_float_data(self):
        return self.data

    def __eq__(self, other):
        if not isinstance(other, FloatGrid):
            return NotImplemented
        return self.get_float_data() == other.get_float_data()

    def __lt__(self, other):
        if not isinstance(other, FloatGrid):
            return NotImplemented
        return self.get_float_data() < other.get_float_data()

class GridMap:
    """
    GridMap class
    """

    def __init__(self, width, height, resolution,
                 center_x, center_y, init_val=FloatGrid(0.0)):
        """__init__

        :param width: number of grid for width
        :param height: number of grid for height
        :param resolution: grid resolution [m]
        :param center_x: center x position  [m]
        :param center_y: center y position [m]
        :param init_val: initial value for all grid
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.center_x = center_x
        self.center_y = center_y

        self.left_lower_x = self.center_x - self.width / 2.0 * self.resolution
        self.left_lower_y = self.center_y - self.height / 2.0 * self.resolution

        self.n_data = self.width * self.height
        self.data = [init_val] * self.n_data
        self.data_type = type(init_val)

    def get_value_from_xy_index(self, x_ind, y_ind):
        """get_value_from_xy_index

        when the index is out of grid map area, return None

        :param x_ind: x index
        :param y_ind: y index
        """

        grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)

        if 0 <= grid_ind < self.n_data:
            return self.data[grid_ind]
        else:
            return None

    def get_xy_index_from_xy_pos(self, x_pos, y_pos):
        """get_xy_index_from_xy_pos

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_ind = self.calc_xy_index_from_position(
            x_pos, self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(
            y_pos, self.left_lower_y, self.height)

        return x_ind, y_ind

    def set_value_from_xy_pos(self, x_pos, y_pos, val):
        """set_value_from_xy_pos

        return bool flag, which means setting value is succeeded or not

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: grid value
        """

        x_ind, y_ind = self.get_xy_index_from_xy_pos(x_pos, y_pos)

        if (not x_ind) or (not y_ind):
            return False  # NG

        flag = self.set_value_from_xy_index(x_ind, y_ind, val)

        return flag

    def set_value_from_xy_index(self, x_ind, y_ind, val):
        """set_value_from_xy_index

        return bool flag, which means setting value is succeeded or not

        :param x_ind: x index
        :param y_ind: y index
        :param val: grid value
        """

        if (x_ind is None) or (y_ind is None):
            return False, False

        grid_ind = int(y_ind * self.width + x_ind)

        if 0 <= grid_ind < self.n_data and isinstance(val, self.data_type):
            self.data[grid_ind] = val
            return True  # OK
        else:
            return False  # NG

    def set_value_from_polygon(self, pol_x, pol_y, val, inside=True):
        """set_value_from_polygon

        Setting value inside or outside polygon

        :param pol_x: x position list for a polygon
        :param pol_y: y position list for a polygon
        :param val: grid value
        :param inside: setting data inside or outside
        """

        # making ring polygon
        if (pol_x[0] != pol_x[-1]) or (pol_y[0] != pol_y[-1]):
            np.append(pol_x, pol_x[0])
            np.append(pol_y, pol_y[0])

        # setting value for all grid
        for x_ind in range(self.width):
            for y_ind in range(self.height):
                x_pos, y_pos = self.calc_grid_central_xy_position_from_xy_index(
                    x_ind, y_ind)

                flag = self.check_inside_polygon(x_pos, y_pos, pol_x, pol_y)

                if flag is inside:
                    self.set_value_from_xy_index(x_ind, y_ind, val)

    def calc_grid_index_from_xy_index(self, x_ind, y_ind):
        grid_ind = int(y_ind * self.width + x_ind)
        return grid_ind

    def calc_xy_index_from_grid_index(self, grid_ind):
        y_ind, x_ind = divmod(grid_ind, self.width)
        return x_ind, y_ind

    def calc_grid_index_from_xy_pos(self, x_pos, y_pos):
        """get_xy_index_from_xy_pos

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_ind = self.calc_xy_index_from_position(x_pos, self.left_lower_x, self.width)
        y_ind = self.calc_xy_index_from_position(y_pos, self.left_lower_y, self.height)

        return self.calc_grid_index_from_xy_index(x_ind, y_ind)

    def calc_grid_central_xy_position_from_grid_index(self, grid_ind):
        x_ind, y_ind = self.calc_xy_index_from_grid_index(grid_ind)
        return self.calc_grid_central_xy_position_from_xy_index(x_ind, y_ind)

    def calc_grid_central_xy_position_from_xy_index(self, x_ind, y_ind):
        x_pos = self.calc_grid_central_xy_position_from_index(
            x_ind, self.left_lower_x)
        y_pos = self.calc_grid_central_xy_position_from_index(
            y_ind, self.left_lower_y)

        return x_pos, y_pos

    def calc_grid_central_xy_position_from_index(self, index, lower_pos):
        return lower_pos + index * self.resolution + self.resolution / 2.0

    def calc_xy_index_from_position(self, pos, lower_pos, max_index):
        ind = int(np.floor((pos - lower_pos) / self.resolution))
        if 0 <= ind <= max_index:
            return ind
        else:
            return None

    def check_occupied_from_xy_index(self, x_ind, y_ind, occupied_val):

        val = self.get_value_from_xy_index(x_ind, y_ind)

        if val is None or val >= occupied_val:
            return True
        else:
            return False

    def expand_grid(self, occupied_val=FloatGrid(1.0)):
        x_inds, y_inds, values = [], [], []

        for ix in range(self.width):
            for iy in range(self.height):
                if self.check_occupied_from_xy_index(ix, iy, occupied_val):
                    x_inds.append(ix)
                    y_inds.append(iy)
                    values.append(self.get_value_from_xy_index(ix, iy))

        for (ix, iy, value) in zip(x_inds, y_inds, values):
            self.set_value_from_xy_index(ix + 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy + 1, val=value)
            self.set_value_from_xy_index(ix + 1, iy + 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy - 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy - 1, val=value)

    @staticmethod
    def check_inside_polygon(iox, ioy, x, y):

        n_point = len(x) - 1
        inside = False
        for i1 in range(n_point):
            i2 = (i1 + 1) % (n_point + 1)

            if x[i1] >= x[i2]:
                min_x, max_x = x[i2], x[i1]
            else:
                min_x, max_x = x[i1], x[i2]
            if not min_x <= iox < max_x:
                continue

            tmp1 = (y[i2] - y[i1]) / (x[i2] - x[i1])
            if (y[i1] + tmp1 * (iox - x[i1]) - ioy) > 0.0:
                inside = not inside

        return inside

    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("resolution:", self.resolution)
        print("center_x:", self.center_x)
        print("center_y:", self.center_y)
        print("left_lower_x:", self.left_lower_x)
        print("left_lower_y:", self.left_lower_y)
        print("n_data:", self.n_data)

    def plot_grid_map(self, ax=None):
        float_data_array = np.array([d.get_float_data() for d in self.data])
        grid_data = np.reshape(float_data_array, (self.height, self.width))
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(grid_data, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.axis("equal")

        return heat_map

class NDTMap:
    """
    Normal Distribution Transform (NDT) map class

    :param ox: obstacle x position list
    :param oy: obstacle y position list
    :param resolution: grid resolution [m]
    """

    class NDTGrid:
        """
        NDT grid
        """

        def __init__(self):
            #: Number of points in the NDTGrid grid
            self.n_points = 0
            #: Mean x position of points in the NDTGrid cell
            self.mean_x = None
            #: Mean y position of points in the NDTGrid cell
            self.mean_y = None
            #: Center x position of the NDT grid
            self.center_grid_x = None
            #: Center y position of the NDT grid
            self.center_grid_y = None
            #: Covariance matrix of the NDT grid
            self.covariance = None  # 协方差矩阵
            #: Eigen vectors of the NDT grid
            self.eig_vec = None  # 特征向量
            #: Eigen values of the NDT grid
            self.eig_values = None  # 特征值

    def __init__(self, ox, oy, resolution):
        #: Minimum number of points in the NDT grid
        self.min_n_points = 3
        #: Resolution of the NDT grid [m]
        self.resolution = resolution
        width = int((max(ox) - min(ox))/resolution) + 3  # rounding up + right and left margin
        height = int((max(oy) - min(oy))/resolution) + 3
        center_x = np.mean(ox)
        center_y = np.mean(oy)
        self.ox = ox
        self.oy = oy
        self.grid_map = GridMap(width, height, resolution, center_x, center_y, self.NDTGrid())
        #: NDT grid index map
        self.grid_index_map = self._create_grid_index_map(ox, oy)

        #: NDT grid map. Each grid contains NDTGrid object
        self._construct_grid_map(center_x, center_y, height, ox, oy, resolution, width)

    def _construct_grid_map(self, center_x, center_y, height, ox, oy, resolution, width):
        # self.grid_map = GridMap(width, height, resolution, center_x, center_y, self.NDTGrid())
        for grid_index, inds in self.grid_index_map.items():
            ndt = self.NDTGrid()
            ndt.n_points = len(inds)
            if ndt.n_points >= self.min_n_points:
                ndt.mean_x = np.mean(ox[inds])
                ndt.mean_y = np.mean(oy[inds])
                ndt.center_grid_x, ndt.center_grid_y = \
                    self.grid_map.calc_grid_central_xy_position_from_grid_index(grid_index)
                ndt.covariance = np.cov(ox[inds], oy[inds])
                ndt.eig_values, ndt.eig_vec = np.linalg.eig(ndt.covariance)
                self.grid_map.data[grid_index] = ndt

    def _create_grid_index_map(self, ox, oy):
        grid_index_map = defaultdict(list)
        for i in range(len(ox)):
            grid_index = self.grid_map.calc_grid_index_from_xy_pos(ox[i], oy[i])
            grid_index_map[grid_index].append(i)
        return grid_index_map


def create_dummy_observation_data():
    ox = []
    oy = []
    # left corridor
    for y in range(-50, 50):  # 
        ox.append(-20.0)
        oy.append(y)
    # right corridor 1
    for y in range(-50, 0):
        ox.append(20.0)
        oy.append(y)
    # right corridor 2
    for x in range(20, 50):
        ox.append(x)
        oy.append(0)
    # right corridor 3
    for x in range(20, 50):
        ox.append(x)
        oy.append(x/2.0+10)
    # right corridor 4
    for y in range(20, 50):
        ox.append(20)
        oy.append(y)
    ox = np.array(ox)
    oy = np.array(oy)
    # Adding random noize
    ox += np.random.rand(len(ox)) * 5.0
    oy += np.random.rand(len(ox)) * 5.0
    return ox, oy


def main():
    print(__file__ + " start!!")

    ox, oy = create_dummy_observation_data()  # 模拟点云数据
    grid_resolution = 10.0  # 栅格分辨率，对应户外
    ndt_map = NDTMap(ox, oy, grid_resolution)  # 初始化NDT map

    # plot raw observation
    plt.plot(ox, oy, ".r")
    plt.xticks(np.arange(-60, 60, 10))  # x轴每10个单位一个刻度
    plt.yticks(np.arange(-60, 60, 10))  # y轴每0.2个单位一个刻度
    plt.grid(True)
    #plt.grid()
    # plot grid clustering
    [plt.plot(ox[inds], oy[inds], "x") for inds in ndt_map.grid_index_map.values()]

    # plot ndt grid map
    [plot_covariance_ellipse(ndt.mean_x, ndt.mean_y, ndt.covariance, color="-k") for ndt in ndt_map.grid_map.data if ndt.n_points > 0]

    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()

"""
Object shape recognition with L-shape fitting
Reference:
- Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners -
The Robotics Institute Carnegie Mellon University
https://www.ri.cmu.edu/publications/
efficient-l-shape-fitting-for-vehicle-detection-using-laser-scanners/

"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import itertools
from enum import Enum

from utils.angle import rot_mat_2d

from simulator import VehicleSimulator, LidarSimulator

show_animation = True

from utils.plot import get_frame_as_array
import imageio
save_gif = False
save_path = './rect.gif'


class LShapeFitting:
    """
    LShapeFitting class. You can use this class by initializing the class and
    changing the parameters, and then calling the fitting method.

    """

    class Criteria(Enum):
        AREA = 1
        CLOSENESS = 2
        VARIANCE = 3

    def __init__(self):
        """
        Default parameter settings
        """
        #: Fitting criteria parameter
        self.criteria = self.Criteria.VARIANCE
        #: Minimum distance for closeness criteria parameter [m]
        self.min_dist_of_closeness_criteria = 0.01
        #: Angle difference parameter [deg]
        self.d_theta_deg_for_search = 1.0
        #: Range segmentation parameter [m]
        self.R0 = 3.0
        #: Range segmentation parameter [m]
        self.Rd = 0.001

    def fitting(self, ox, oy):
        """
        Fitting L-shape model to object points

        Parameters
        ----------
        ox : x positions of range points from an object
        oy : y positions of range points from an object

        Returns
        -------
        rects: Fitting rectangles
        id_sets: id sets of each cluster

        """
        # step1: Adaptive Range Segmentation
        id_sets = self._adoptive_range_segmentation(ox, oy)

        # step2 Rectangle search
        rects = []
        for ids in id_sets:  # for each cluster
            cx = [ox[i] for i in range(len(ox)) if i in ids]
            cy = [oy[i] for i in range(len(oy)) if i in ids]
            rects.append(self._rectangle_search(cx, cy))

        return rects, id_sets

    @staticmethod
    def _calc_area_criterion(c1, c2):
        c1_max, c1_min, c2_max, c2_min = LShapeFitting._find_min_max(c1, c2)
        alpha = -(c1_max - c1_min) * (c2_max - c2_min)
        return alpha

    def _calc_closeness_criterion(self, c1, c2):
        c1_max, c1_min, c2_max, c2_min = LShapeFitting._find_min_max(c1, c2)

        # Vectorization
        d1 = np.minimum(c1_max - c1, c1 - c1_min)
        d2 = np.minimum(c2_max - c2, c2 - c2_min)
        d = np.maximum(np.minimum(d1, d2), self.min_dist_of_closeness_criteria)
        beta = (1.0 / d).sum()

        return beta

    @staticmethod
    def _calc_variance_criterion(c1, c2):
        c1_max, c1_min, c2_max, c2_min = LShapeFitting._find_min_max(c1, c2)

        # Vectorization
        d1 = np.minimum(c1_max - c1, c1 - c1_min)
        d2 = np.minimum(c2_max - c2, c2 - c2_min)
        e1 = d1[d1 < d2]
        e2 = d2[d1 >= d2]
        v1 = - np.var(e1) if len(e1) > 0 else 0.
        v2 = - np.var(e2) if len(e2) > 0 else 0.
        gamma = v1 + v2

        return gamma

    @staticmethod
    def _find_min_max(c1, c2):
        c1_max = max(c1)
        c2_max = max(c2)
        c1_min = min(c1)
        c2_min = min(c2)
        return c1_max, c1_min, c2_max, c2_min

    def _rectangle_search(self, x, y):

        xy = np.array([x, y]).T

        d_theta = np.deg2rad(self.d_theta_deg_for_search)
        min_cost = (-float('inf'), None)
        for theta in np.arange(0.0, np.pi / 2.0 - d_theta, d_theta):

            c = xy @ rot_mat_2d(theta)
            c1 = c[:, 0]
            c2 = c[:, 1]

            # Select criteria
            cost = 0.0
            if self.criteria == self.Criteria.AREA:
                cost = self._calc_area_criterion(c1, c2)
            elif self.criteria == self.Criteria.CLOSENESS:
                cost = self._calc_closeness_criterion(c1, c2)
            elif self.criteria == self.Criteria.VARIANCE:
                cost = self._calc_variance_criterion(c1, c2)

            if min_cost[0] < cost:
                min_cost = (cost, theta)

        # calc best rectangle
        sin_s = np.sin(min_cost[1])
        cos_s = np.cos(min_cost[1])

        c1_s = xy @ np.array([cos_s, sin_s]).T
        c2_s = xy @ np.array([-sin_s, cos_s]).T

        rect = RectangleData()
        rect.a[0] = cos_s
        rect.b[0] = sin_s
        rect.c[0] = min(c1_s)
        rect.a[1] = -sin_s
        rect.b[1] = cos_s
        rect.c[1] = min(c2_s)
        rect.a[2] = cos_s
        rect.b[2] = sin_s
        rect.c[2] = max(c1_s)
        rect.a[3] = -sin_s
        rect.b[3] = cos_s
        rect.c[3] = max(c2_s)

        return rect

    def _adoptive_range_segmentation(self, ox, oy):

        # Setup initial cluster
        segment_list = []
        for i, _ in enumerate(ox):
            c = set()
            r = self.R0 + self.Rd * np.linalg.norm([ox[i], oy[i]])
            for j, _ in enumerate(ox):
                d = np.hypot(ox[i] - ox[j], oy[i] - oy[j])
                if d <= r:
                    c.add(j)
            segment_list.append(c)

        # Merge cluster
        while True:
            no_change = True
            for (c1, c2) in list(itertools.permutations(range(len(segment_list)), 2)):
                if segment_list[c1] & segment_list[c2]:
                    segment_list[c1] = (segment_list[c1] | segment_list.pop(c2))
                    no_change = False
                    break
            if no_change:
                break

        return segment_list


class RectangleData:

    def __init__(self):
        # 初始化长度为4的列表a，所有元素初始值为None
        self.a = [None] * 4
        # 初始化长度为4的列表b，所有元素初始值为None
        self.b = [None] * 4
        # 初始化长度为4的列表c，所有元素初始值为None
        self.c = [None] * 4

        # 初始化长度为5的列表rect_c_x，用于存储矩形c的x坐标，所有元素初始值为None
        self.rect_c_x = [None] * 5
        # 初始化长度为5的列表rect_c_y，用于存储矩形c的y坐标，所有元素初始值为None
        self.rect_c_y = [None] * 5

    def plot(self):
        self.calc_rect_contour()
        plt.plot(self.rect_c_x, self.rect_c_y, "-r")

    def calc_rect_contour(self):

        self.rect_c_x[0], self.rect_c_y[0] = self.calc_cross_point(
            self.a[0:2], self.b[0:2], self.c[0:2])
        self.rect_c_x[1], self.rect_c_y[1] = self.calc_cross_point(
            self.a[1:3], self.b[1:3], self.c[1:3])
        self.rect_c_x[2], self.rect_c_y[2] = self.calc_cross_point(
            self.a[2:4], self.b[2:4], self.c[2:4])
        self.rect_c_x[3], self.rect_c_y[3] = self.calc_cross_point(
            [self.a[3], self.a[0]], [self.b[3], self.b[0]], [self.c[3], self.c[0]])
        self.rect_c_x[4], self.rect_c_y[4] = self.rect_c_x[0], self.rect_c_y[0]

    @staticmethod
    def calc_cross_point(a, b, c):
        x = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0])
        y = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0])
        return x, y


def main():

    # simulation parameters
    sim_time = 30.0  # simulation time
    dt = 0.2  # time tick

    angle_resolution = np.deg2rad(2.0)  # sensor angle resolution
    # x,y,theta, v, max_v, length, width,
    v1 = VehicleSimulator(-10.0, 0.0, np.deg2rad(90.0),
                          0.0, 50.0 / 3.6, 3.0, 5.0)
    v2 = VehicleSimulator(20.0, 10.0, np.deg2rad(180.0),
                          0.0, 50.0 / 3.6, 4.0, 10.0)

    l_shape_fitting = LShapeFitting()
    lidar_sim = LidarSimulator()

    paused = False

    def on_key_press(event):
        nonlocal paused
        if event.key == ' ':  # 空格键切换暂停状态
            paused = not paused
        elif event.key == 'escape':  # ESC键退出
            sys.exit(0)

    time = 0.0

    # save gif
    frames = []
    fig = None
    ax = None
    if save_gif:
        fig = plt.figure(figsize=(8, 8), dpi=80)
        ax = fig.add_subplot(111)

    while time <= sim_time:

        if not paused:
            time += dt
            v1.update(dt, 0.1, 0.0)  # 加速度和角速度
            v2.update(dt, 0.1, -0.05)

        ox, oy = lidar_sim.get_observation_points([v1, v2], angle_resolution)

        rects, id_sets = l_shape_fitting.fitting(ox, oy)

        if show_animation:  # pragma: no cover
            if show_animation:
                plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)
                if save_gif:  # 清空画布
                    ax.cla()
                else:
                    plt.cla()

            if paused:
                plt.text(0.05, 0.95, "PAUSED", transform=plt.gca().transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
            plt.axis("equal")
            plt.plot(0.0, 0.0, "*r")
            v1.plot()
            v2.plot()

            # Plot range observation
            for ids in id_sets:
                x = [ox[i] for i in range(len(ox)) if i in ids]
                y = [oy[i] for i in range(len(ox)) if i in ids]

                for (ix, iy) in zip(x, y):
                    plt.plot([0.0, ix], [0.0, iy], "-og")

                plt.plot([ox[i] for i in range(len(ox)) if i in ids],
                         [oy[i] for i in range(len(ox)) if i in ids],
                         "o")
            #''' 检测结果 逆推的矩形心态
            for rect in rects:
                rect.plot()
            #'''
            if save_gif:
                fig.canvas.draw()
                frame = get_frame_as_array(fig)
                frames.append(frame)
            if show_animation:  # and not save_gif:
                plt.pause(0.1)

    if save_gif and len(frames) > 0:
        imageio.mimsave(save_path, frames, fps=10, loop=0)  # 保存为GIF
        print(f"GIF saved to {save_path}")
        # plt.pause(0.1)

    print("Done")


if __name__ == '__main__':
    main()

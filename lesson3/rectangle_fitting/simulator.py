import sys
import pathlib
# sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import math
import random

from utils.angle import rot_mat_2d


class VehicleSimulator:

    def __init__(self, i_x, i_y, i_yaw, i_v, max_v, w, L):

        self.x = i_x  # 车辆当前x坐标
        self.y = i_y  # 车辆当前y坐标
        self.yaw = i_yaw  # 车辆当前偏航角(弧度)
        self.v = i_v  # 车辆当前速度
        self.max_v = max_v  # 车辆最大速度限制
        self.W = w  # 车辆宽度
        self.L = L  # 车辆长度
        self._calc_vehicle_contour()  # 计算车辆轮廓点

    def update(self, dt, a, omega):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += omega * dt
        self.v += a * dt  # 加速度
        if self.v >= self.max_v:
            self.v = self.max_v

    def plot(self):
        plt.plot(self.x, self.y, ".b")

        # convert global coordinate
        gx, gy = self.calc_global_contour()
        plt.plot(gx, gy, "--b")

    def calc_global_contour(self):
        gxy = np.stack([self.vc_x, self.vc_y]).T @ rot_mat_2d(self.yaw)
        gx = gxy[:, 0] + self.x
        gy = gxy[:, 1] + self.y

        return gx, gy

    def _calc_vehicle_contour(self):

        self.vc_x = []
        self.vc_y = []
        # 4个顶点
        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(-self.W / 2.0)

        self.vc_x.append(-self.L / 2.0)
        self.vc_y.append(-self.W / 2.0)

        self.vc_x.append(-self.L / 2.0)
        self.vc_y.append(self.W / 2.0)

        self.vc_x.append(self.L / 2.0)
        self.vc_y.append(self.W / 2.0)
        # 线性插值，增加车辆轮廓密度。
        self.vc_x, self.vc_y = self._interpolate(self.vc_x, self.vc_y)

    @staticmethod
    def _interpolate(x, y):
        rx, ry = [], []
        d_theta = 0.05  # 插值步长
        for i in range(len(x) - 1):
            rx.extend([(1.0 - theta) * x[i] + theta * x[i + 1]
                       for theta in np.arange(0.0, 1.0, d_theta)])
            ry.extend([(1.0 - theta) * y[i] + theta * y[i + 1]
                       for theta in np.arange(0.0, 1.0, d_theta)])

        rx.extend([(1.0 - theta) * x[len(x) - 1] + theta * x[1]
                   for theta in np.arange(0.0, 1.0, d_theta)])
        ry.extend([(1.0 - theta) * y[len(y) - 1] + theta * y[1]
                   for theta in np.arange(0.0, 1.0, d_theta)])

        return rx, ry


class LidarSimulator:

    def __init__(self):
        self.range_noise = 0.05

    def get_observation_points(self, v_list, angle_resolution):
        x, y, angle, r = [], [], [], []

        # store all points
        for v in v_list:  # 遍历每个车辆

            gx, gy = v.calc_global_contour()  # 当前车的全局轮廓点

            for vx, vy in zip(gx, gy):
                v_angle = math.atan2(vy, vx)  # 当前点的角度
                # 计算当前点的距离，带噪
                vr = np.hypot(vx, vy) * random.uniform(1.0 - self.range_noise,
                                                       1.0 + self.range_noise)

                x.append(vx)  # 坐标
                y.append(vy)
                angle.append(v_angle)  # 角度
                r.append(vr)  # 距离

        # ray casting filter
        rx, ry = self.ray_casting_filter(angle, r, angle_resolution)

        return rx, ry

    @staticmethod
    def ray_casting_filter(theta_l, range_l, angle_resolution):
        rx, ry = [], []
        # 初始化一个角度范围的列表，长度为 360/角分辨率+1，，全为无穷大
        range_db = [float("inf") for _ in range(
            int(np.floor((np.pi * 2.0) / angle_resolution)) + 1)]
        # 遍历所有（车辆采样点）角度
        for i in range(len(theta_l)):
            angle_id = int(round(theta_l[i] / angle_resolution))
            # 存为最近距离
            if range_db[angle_id] > range_l[i]:
                range_db[angle_id] = range_l[i]
        # 遍历所有角度范围，计算过滤后的点坐标。
        for i in range(len(range_db)):
            t = i * angle_resolution
            if range_db[i] != float("inf"):
                rx.append(range_db[i] * np.cos(t))
                ry.append(range_db[i] * np.sin(t))

        return rx, ry

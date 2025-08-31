"""

Ray casting 2D grid map example

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import numpy as np
import matplotlib.pyplot as plt

from PythonRobotics.ArmNavigation.arm_obstacle_navigation.arm_obstacle_navigation import obstacles

EXTEND_AREA = 10.0

show_animation = True


def calc_grid_map_config(ox, oy, xyreso):
    """
    计算栅格地图的配置信息，包括地图的边界和尺寸。

    参数:
    ox (list): 障碍物的 x 坐标列表
    oy (list): 障碍物的 y 坐标列表
    xyreso (float): 栅格地图的分辨率

    返回:
    tuple: 包含地图的最小 x 坐标、最小 y 坐标、最大 x 坐标、最大 y 坐标、
           x 方向的栅格数量和 y 方向的栅格数量
    """
    # 计算地图的最小 x 坐标，通过障碍物最小 x 坐标减去扩展区域的一半并取整
    minx = round(min(ox) - EXTEND_AREA / 2.0)
    # 计算地图的最小 y 坐标，通过障碍物最小 y 坐标减去扩展区域的一半并取整
    miny = round(min(oy) - EXTEND_AREA / 2.0)
    # 计算地图的最大 x 坐标，通过障碍物最大 x 坐标加上扩展区域的一半并取整
    maxx = round(max(ox) + EXTEND_AREA / 2.0)
    # 计算地图的最大 y 坐标，通过障碍物最大 y 坐标加上扩展区域的一半并取整
    maxy = round(max(oy) + EXTEND_AREA / 2.0)
    # 计算 x 方向的栅格数量，通过地图 x 方向长度除以分辨率并取整
    xw = int(round((maxx - minx) / xyreso))
    # 计算 y 方向的栅格数量，通过地图 y 方向长度除以分辨率并取整
    yw = int(round((maxy - miny) / xyreso))

    return minx, miny, maxx, maxy, xw, yw


class precastDB:

    # 初始化类
    def __init__(self):
        # 初始化x坐标
        self.px = 0.0
        # 初始化y坐标
        self.py = 0.0
        # 初始化直径
        self.d = 0.0
        # 初始化角度
        self.angle = 0.0
        # 初始化x索引
        self.ix = 0
        # 初始化y索引
        self.iy = 0

    # 返回类的字符串表示
    def __str__(self):
        # 返回x坐标、y坐标、直径、角度的字符串表示
        return str(self.px) + "," + str(self.py) + "," + str(self.d) + "," + str(self.angle)


# 定义一个函数，用于计算从0到2π的角度
def atan_zero_to_twopi(y, x):
    # 使用math库中的atan2函数计算角度
    angle = math.atan2(y, x)
    # 如果角度小于0，则加上2π
    if angle < 0.0:
        angle += math.pi * 2.0

    # 返回角度
    return angle


def pre_casting(minx, miny, xw, yw, xyreso, yawreso):

    # 创建一个空的列表，用于存储预计算的点
    precast = [[] for i in range(int(round((math.pi * 2.0) / yawreso)) + 1)]

    # 遍历x和y方向上的点
    for ix in range(xw):
        for iy in range(yw):
            # 计算当前点的x和y坐标
            px = ix * xyreso + minx
            py = iy * xyreso + miny

            # 计算当前点到原点的距离
            d = math.hypot(px, py)
            # 计算当前点的角度
            angle = atan_zero_to_twopi(py, px)
            # 计算当前点的角度id
            angleid = int(math.floor(angle / yawreso))

            # 创建一个预计算的点对象
            pc = precastDB()

            # 设置预计算的点的属性
            pc.px = px
            pc.py = py
            pc.d = d
            pc.ix = ix
            pc.iy = iy
            pc.angle = angle

            # 将预计算的点添加到对应的列表中
            precast[angleid].append(pc)

    # 返回预计算的点列表
    return precast


def generate_ray_casting_grid_map(ox, oy, xyreso, yawreso):

    # 计算网格地图的配置参数
    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso)

    # 初始化网格地图
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    # 预先计算射线投射
    precast = pre_casting(minx, miny, xw, yw, xyreso, yawreso)

    # 遍历所有障碍物点
    for (x, y) in zip(ox, oy):

        # 计算障碍物点到原点的距离
        d = math.hypot(x, y)
        # 计算障碍物点的角度
        angle = atan_zero_to_twopi(y, x)
        # 计算障碍物点的角度id
        angleid = int(math.floor(angle / yawreso))

        # 获取预先计算的射线投射列表
        gridlist = precast[angleid]

        # 计算障碍物点的网格坐标
        ix = int(round((x - minx) / xyreso))
        iy = int(round((y - miny) / xyreso))

        # 遍历射线投射列表
        for grid in gridlist:
            # 如果射线投射的距离大于障碍物点到原点的距离，则将网格地图中对应的点设置为0.5
            if grid.d > d:
                pmap[grid.ix][grid.iy] = 0.5

        # 将障碍物点对应的网格地图中对应的点设置为1.0
        pmap[ix][iy] = 1.0

    # 返回网格地图和网格地图的配置参数
    return pmap, minx, maxx, miny, maxy, xyreso


def draw_heatmap(data, minx, maxx, miny, maxy, xyreso):
    x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso),
                    slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]
    plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues)
    plt.axis("equal")

# 新增：空格键暂停回调函数
def toggle_pause(event):
    global is_paused
    if event.key == " ":  # 检测空格键
        is_paused = not is_paused  # 切换暂停状态
        print("暂停" if is_paused else "继续")

def main():
    print(__file__ + " start!!")
    global is_paused  # 使用全局暂停标志
    is_paused = False

    xyreso = 0.25  # x-y grid resolution [m]
    yawreso = np.deg2rad(5)  # yaw angle resolution [rad]

    for i in range(4):
        obstacles = 5
        ox = (np.random.rand(obstacles) - 0.5) * 10.0  # 生成4个随机数，范围在-5到5之间
        oy = (np.random.rand(obstacles) - 0.5) * 10.0  # 生成4个随机数，范围在-5到5之间, 用于生成障碍物
        pmap, minx, maxx, miny, maxy, xyreso = generate_ray_casting_grid_map(
            ox, oy, xyreso, yawreso)  # 生成栅格地图

        if show_animation:  # pragma: no cover
            plt.cla()  # 清空当前图像
            plt.gcf().canvas.mpl_connect("key_press_event", toggle_pause)
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])  # 按下esc键退出
            draw_heatmap(pmap, minx, maxx, miny, maxy, xyreso)  # 绘制热力图
            plt.plot(ox, oy, "xr", markersize=10)  # 绘制障碍物
            plt.plot(0.0, 0.0, "ob")  # 绘制起点
            plt.pause(1.0)  # 暂停1秒
            # plt.show()
            while is_paused:  # 检查是否暂停
                plt.pause(0.1)  # 暂停时暂停0.1秒

if __name__ == '__main__':
    main()

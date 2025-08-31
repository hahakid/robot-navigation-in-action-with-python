"""
Ray casting 2D grid map example
"""

import math
import numpy as np
import matplotlib.pyplot as plt


from utils.plot import get_frame_as_array
import imageio

EXTEND_AREA = 10.0

show_animation = True
save_gif = True  # 是否保存GIF
output_path = "ray_casting_animation.gif"  # GIF输出路径

def calc_grid_map_config(ox, oy, xyreso):
    minx = round(min(ox) - EXTEND_AREA / 2.0)
    miny = round(min(oy) - EXTEND_AREA / 2.0)
    maxx = round(max(ox) + EXTEND_AREA / 2.0)
    maxy = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))

    return minx, miny, maxx, maxy, xw, yw


class precastDB:

    def __init__(self):
        self.px = 0.0
        self.py = 0.0
        self.d = 0.0
        self.angle = 0.0
        self.ix = 0
        self.iy = 0

    def __str__(self):
        return str(self.px) + "," + str(self.py) + "," + str(self.d) + "," + str(self.angle)


def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0

    return angle


def pre_casting(minx, miny, xw, yw, xyreso, yawreso):

    precast = [[] for i in range(int(round((math.pi * 2.0) / yawreso)) + 1)]

    for ix in range(xw):
        for iy in range(yw):
            px = ix * xyreso + minx
            py = iy * xyreso + miny

            d = math.hypot(px, py)
            angle = atan_zero_to_twopi(py, px)
            angleid = int(math.floor(angle / yawreso))

            pc = precastDB()

            pc.px = px
            pc.py = py
            pc.d = d
            pc.ix = ix
            pc.iy = iy
            pc.angle = angle

            precast[angleid].append(pc)

    return precast


def generate_ray_casting_grid_map(ox, oy, xyreso, yawreso):

    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso)

    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    precast = pre_casting(minx, miny, xw, yw, xyreso, yawreso)

    for (x, y) in zip(ox, oy):

        d = math.hypot(x, y)
        angle = atan_zero_to_twopi(y, x)
        angleid = int(math.floor(angle / yawreso))

        gridlist = precast[angleid]

        ix = int(round((x - minx) / xyreso))
        iy = int(round((y - miny) / xyreso))

        for grid in gridlist:
            if grid.d > d:
                pmap[grid.ix][grid.iy] = 0.5

        pmap[ix][iy] = 1.0

    return pmap, minx, maxx, miny, maxy, xyreso


def draw_heatmap(data, minx, maxx, miny, maxy, xyreso, ax=None):
    if ax is None:
        ax = plt.gca()

    x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso),
                    slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]
    im = ax.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues)
    plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues)
    plt.axis("equal")
    return im  # 返回图像对象


def main():
    global ax
    print(__file__ + " start!!")
    obstacles = 5
    xyreso = 0.25  # x-y grid resolution [m]
    # yawreso = np.deg2rad(10.0)  # yaw angle resolution [rad]
    ox = (np.random.rand(obstacles) - 0.5) * 10.0
    oy = (np.random.rand(obstacles) - 0.5) * 10.0

    if save_gif:
        frames = []
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)


    for i in range(10):  # 随机生成五帧
        yawreso = np.deg2rad(10 - i)  # yaw angle resolution [rad], 不断增加传感器的角分辨率

        pmap, minx, maxx, miny, maxy, xyreso = generate_ray_casting_grid_map(
            ox, oy, xyreso, yawreso)

        if show_animation or save_gif:  # pragma: no cover
            if not save_gif:
                plt.figure()
                ax = plt.gca()
            else:
                ax.clear()

            ax.set_title(f"Ray Casting Grid Map - Iteration {i + 1}")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            draw_heatmap(pmap, minx, maxx, miny, maxy, xyreso, ax)
            ax.plot(ox, oy, "xr", markersize=10)  # 绘制障碍物
            ax.plot(0.0, 0.0, "ob", markersize=10)  # 绘制原点(传感器位置)

            if show_animation:
                plt.draw()
                plt.pause(1.0)
            if save_gif:
                fig.canvas.draw()
                frame = get_frame_as_array(fig)
                frames.append(frame)

    if save_gif:
        imageio.mimsave(output_path, frames, fps=1, loop=0)  # 保存为GIF
        print(f"GIF saved to {output_path}")

    if show_animation and not save_gif:  # pragma: no cover
        plt.show()


if __name__ == '__main__':
    main()

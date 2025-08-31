"""

2D gaussian grid map sample


"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

EXTEND_AREA = 10.0  # [m] grid map extention length

show_animation = True


def generate_gaussian_grid_map(ox, oy, xyreso, std):

    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso)

    gmap = [[0.0 for i in range(yw)] for i in range(xw)]  # 初始化 全0

    for ix in range(xw):
        for iy in range(yw):

            x = ix * xyreso + minx
            y = iy * xyreso + miny

            # Search minimum distance
            mindis = float("inf")
            for (iox, ioy) in zip(ox, oy):
                d = math.hypot(iox - x, ioy - y)
                if mindis >= d:
                    mindis = d
            '''
            1.0 - CDF：表示 " 距离大于mindis的概率 "，其特性是：
                当网格离障碍物越近（mindis越小），该值越接近 1.0（颜色越深）
                当网格离障碍物越远（mindis越大），该值越接近 0.0（颜色越浅）
            '''
            pdf = (1.0 - norm.cdf(mindis, 0.0, std))  # mean=0, std=std
            gmap[ix][iy] = pdf

    return gmap, minx, maxx, miny, maxy


def calc_grid_map_config(ox, oy, xyreso):
    minx = round(min(ox) - EXTEND_AREA / 2.0)
    miny = round(min(oy) - EXTEND_AREA / 2.0)
    maxx = round(max(ox) + EXTEND_AREA / 2.0)
    maxy = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))

    return minx, miny, maxx, maxy, xw, yw


def draw_heatmap(data, minx, maxx, miny, maxy, xyreso):

    # 创建网格，x和y的范围分别为[minx, maxx]和[miny, maxy]，步长为xyreso
    x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso),
                    slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]
    # 绘制热图，数据为data，最大值为1.0，颜色映射为Blues
    plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues)
    # 设置坐标轴比例相等
    plt.axis("equal")


def main():
    print(__file__ + " start!!")

    xyreso = 0.5  # xy grid resolution
    STD = 5  # standard diviation for gaussian distribution
    num_obs = 4
    for i in range(5):
        ox = (np.random.rand(num_obs) - 0.5) * 10.0
        oy = (np.random.rand(num_obs) - 0.5) * 10.0
        for s in range(STD):
            gmap, minx, maxx, miny, maxy = generate_gaussian_grid_map(ox, oy, xyreso, float(s))

            if show_animation:  # pragma: no cover
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                draw_heatmap(gmap, minx, maxx, miny, maxy, xyreso)
                plt.plot(ox, oy, "xr")
                plt.plot(0.0, 0.0, "og")
                # plt.pause(1.0)
                plt.show()


def visualize_gaussian_cdf(std=5.0):
    # Generate distance range (0 to 3x standard deviation, covering main distribution area)
    distances = np.linspace(0, 3 * std, 1000)

    # Calculate CDF: P(distance ≤ d)
    cdf = norm.cdf(distances, loc=0.0, scale=std)

    # Calculate 1-CDF: P(distance > d), which is the pdf in our code
    one_minus_cdf = 1.0 - cdf

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances, cdf, label='CDF: P(distance ≤ d)', color='blue', linewidth=2)
    plt.plot(distances, one_minus_cdf, label='1-CDF: P(distance > d)', color='red', linewidth=2)

    # Add reference lines
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=std, color='green', linestyle='--', alpha=0.5, label=f'Standard deviation = {std}')

    # Set axis labels and title
    plt.xlabel('Distance to obstacle (mindis)', fontsize=22)
    plt.ylabel('Probability value', fontsize=22)
    plt.title('Relationship between Gaussian CDF and 1-CDF', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == '__main__':
    # visualize_gaussian_cdf(std=5.0)
    main()

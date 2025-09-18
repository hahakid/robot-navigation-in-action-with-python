"""
基于势场法的路径规划器
参考资料:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

势场法路径规划的基本原理：
- 目标点对机器人产生"引力"，吸引机器人向其移动
- 障碍物对机器人产生"斥力"，排斥机器人远离它们
- 机器人的运动方向由引力和斥力的合力决定
"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from utils.plot import get_frame_as_array
import imageio, random
show_animation = True
save_gif = True  # 是否保存GIF
output_path = "PF.gif"  # GIF输出路径


# 算法参数设置
KP = 5.0  # 引力势场增益，控制引力大小
ETA = 100.0  # 斥力势场增益，控制斥力大小
AREA_WIDTH = 30.0  # 势场计算区域宽度 [米]，超出此范围不计算势场

# 用于检测路径振荡的历史位置数量
OSCILLATIONS_DETECTION_LENGTH = 5

show_animation = True  # 是否显示动画演示


def calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy):
    """
    计算整个规划空间的势场分布
    参数:
        gx, gy: 目标点坐标 [米]
        ox, oy: 障碍物坐标列表 [米]
        reso: 势场网格分辨率 [米]，即每个网格单元的实际尺寸
        rr: 机器人半径 [米]
        sx, sy: 起点坐标 [米]
    返回:
        pmap: 二维数组，存储每个网格点的势场值
        minx, miny: 势场计算区域的最小x、y坐标 [米]
    """
    # 确定势场计算区域的边界（包含起点、终点和所有障碍物，并向外扩展一定范围）
    minx = min(min(ox), sx, gx) - AREA_WIDTH / 2.0
    miny = min(min(oy), sy, gy) - AREA_WIDTH / 2.0
    maxx = max(max(ox), sx, gx) + AREA_WIDTH / 2.0
    maxy = max(max(oy), sy, gy) + AREA_WIDTH / 2.0

    # 计算x和y方向的网格数量
    xw = int(round((maxx - minx) / reso))  # x方向网格数
    yw = int(round((maxy - miny) / reso))  # y方向网格数

    # 初始化势场图（每个网格点的势场值初始化为0）
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    # 遍历每个网格点，计算其势场值（引力势+斥力势）
    for ix in range(xw):
        x = ix * reso + minx  # 网格点对应的实际x坐标
        for iy in range(yw):
            y = iy * reso + miny  # 网格点对应的实际y坐标
            ug = calc_attractive_potential(x, y, gx, gy)  # 计算引力势
            uo = calc_repulsive_potential(x, y, ox, oy, rr)  # 计算斥力势
            uf = ug + uo  # 总势场为引力势与斥力势之和
            pmap[ix][iy] = uf

    return pmap, minx, miny


def calc_attractive_potential(x, y, gx, gy):
    """
    计算目标点对当前位置产生的引力势
    引力势函数：U_g = 0.5 * KP * ||(x,y)-(gx,gy)||²
    （距离目标越远，引力势越大，吸引机器人向目标移动）
    参数:
        x, y: 当前位置坐标 [米]
        gx, gy: 目标点坐标 [米]
    返回:
        引力势值
    """
    # np.hypot计算欧氏距离：sqrt((x-gx)² + (y-gy)²)
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, ox, oy, rr):
    """
    计算障碍物对当前位置产生的斥力势
    斥力势函数：当距离障碍物小于机器人半径时，U_o = 0.5 * ETA * (1/d - 1/rr)²
    （距离障碍物越近，斥力势越大，排斥机器人远离障碍物）
    参数:
        x, y: 当前位置坐标 [米]
        ox, oy: 障碍物坐标列表 [米]
        rr: 机器人半径 [米]，小于此距离时产生斥力
    返回:
        斥力势值
    """
    # 寻找最近的障碍物
    minid = -1  # 最近障碍物的索引
    dmin = float("inf")  # 到最近障碍物的距离
    for i, _ in enumerate(ox):
        d = np.hypot(x - ox[i], y - oy[i])  # 计算当前点到第i个障碍物的距离
        if dmin >= d:
            dmin = d
            minid = i

    # 计算斥力势
    dq = np.hypot(x - ox[minid], y - oy[minid])  # 到最近障碍物的距离

    if dq <= rr:  # 只有当距离小于机器人半径时才产生斥力
        if dq <= 0.1:  # 避免距离为0导致的数值问题
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:  # 距离大于机器人半径时，斥力为0
        return 0.0


def get_motion_model():
    """
    定义机器人的运动模型（可能的移动方向）
    返回:
        motion: 列表，每个元素为[x方向步长, y方向步长]，表示可能的移动方向
    """
    # 8个方向的移动模型（上下左右及四个对角线方向）
    motion = [[1, 0],  # 向右
              [0, 1],  # 向上
              [-1, 0],  # 向左
              [0, -1],  # 向下
              [-1, -1],  # 向左下
              [-1, 1],  # 向左上
              [1, -1],  # 向右下
              [1, 1]]  # 向右上

    return motion


def oscillations_detection(previous_ids, ix, iy):
    """
    检测机器人是否陷入振荡（在相同位置附近往复运动）
    参数:
        previous_ids: 存储最近位置的队列（网格坐标）
        ix, iy: 当前位置的网格坐标
    返回:
        True: 检测到振荡；False: 未检测到振荡
    """
    previous_ids.append((ix, iy))  # 将当前位置加入历史队列

    # 只保留最近的OSCILLATIONS_DETECTION_LENGTH个位置
    if (len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH):
        previous_ids.popleft()

    # 通过集合判断是否有重复位置（若有重复则认为发生振荡）
    previous_ids_set = set()
    for index in previous_ids:
        if index in previous_ids_set:
            return True  # 检测到重复位置，发生振荡
        else:
            previous_ids_set.add(index)
    return False


def potential_field_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    """
    基于势场法进行路径规划
    参数:
        sx, sy: 起点坐标 [米]
        gx, gy: 目标点坐标 [米]
        ox, oy: 障碍物坐标列表 [米]
        reso: 网格分辨率 [米]
        rr: 机器人半径 [米]
    返回:
        rx, ry: 规划出的路径坐标列表 [米]
    """

    # 计算整个空间的势场分布
    pmap, minx, miny = calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy)

    # 初始化路径搜索参数
    d = np.hypot(sx - gx, sy - gy)  # 起点到目标点的初始距离
    # 将起点和目标点的实际坐标转换为网格坐标
    ix = round((sx - minx) / reso)
    iy = round((sy - miny) / reso)
    gix = round((gx - minx) / reso)
    giy = round((gy - miny) / reso)

    fig, ax = None, None
    frames = None

    if show_animation or save_gif:
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        if save_gif:
            frames = []

        ax.grid(True)  # 显示网格
        ax.axis("equal")  # 等比例显示坐标轴
        draw_heatmap(pmap)  # 绘制势场热图
        # 按ESC键退出动画
        ax.plot(ix, iy, "*k", markersize=20)  # 标记起点（黑色星号）
        ax.plot(gix, giy, "*m", markersize=20)  # 标记目标点（紫色星号）

        if save_gif:
            fig.canvas.draw()
            frames.append(get_frame_as_array(fig))
        if show_animation:
            plt.show(block=False)
            plt.pause(0.1)  # 初始显示暂停

        # 存储规划出的路径
        rx, ry = [sx], [sy]
        motion = get_motion_model()  # 获取机器人可能的移动方向
        previous_ids = deque()  # 用于存储历史位置，检测振荡

        oscillation_count = 0
        max_oscillation_retries = 10  # 最多允许尝试逃逸次数

        visited = set()
        # 路径搜索主循环：直到到达目标点附近（距离小于网格分辨率）
        while d >= reso:
            minp = float("inf")  # 最小势场值
            minix, miniy = -1, -1  # 最小势场值对应的网格坐标

            # 遍历所有可能的移动方向，寻找势场值最小的方向
            for i, _ in enumerate(motion):
                inx = int(ix + motion[i][0])  # 移动后的x网格坐标
                iny = int(iy + motion[i][1])  # 移动后的y网格坐标

                # 检查是否超出势场计算区域
                if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
                    p = float("inf")  # 超出区域，势场值设为无穷大
                    print("outside potential!")
                else:
                    p = pmap[inx][iny]  # 获取该位置的势场值

                # 更新最小势场值及对应坐标
                if minp > p:
                    minp = p
                    minix = inx
                    miniy = iny

            # 新的势场值最小的栅格位置
            ix = minix
            iy = miniy
            # 新的势场值最小栅格对应坐标
            xp = ix * reso + minx
            yp = iy * reso + miny
            # 更新到目标点的距离
            d = np.hypot(gx - xp, gy - yp)
            # 将新位置加入路径
            rx.append(xp)
            ry.append(yp)

            visited.add((ix, iy))

            # 检测是否发生振荡，若振荡则退出循环
            if (oscillations_detection(previous_ids, ix, iy)):
                oscillation_count += 1
                print(f"Oscillation detected at ({ix},{iy})!, retry{oscillation_count}")
                if oscillation_count > max_oscillation_retries:
                    print("Max oscillation retries reached. Exiting...")
                    break  # 不是continue，继续，而是直接中断了while循环
                else:
                    step_back_total = 3  # adjustable
                    candidate_moves = get_motion_model()
                    moved = False
                    for _ in range(10):
                        dx, dy = random.choice(candidate_moves)
                        for step in range(1, step_back_total + 1):
                            nx = ix - step * dx
                            ny = iy - step * dy

                            if not (0 <= nx < len(pmap)) and (0 <= ny < len(pmap[0])):
                                break
                            if pmap[nx][ny] == float("inf"):
                                break

                            ix, iy = nx, ny
                            xp = ix * reso + minx
                            yp = iy * reso + miny
                            d = np.hypot(gx - xp, gy - yp)
                            rx.append(xp)
                            ry.append(yp)
                            visited.add((ix, iy))
                            moved = True
                            ax.plot(ix, iy, ".g")
                            if save_gif:
                                fig.canvas.draw()
                                frames.append(get_frame_as_array(fig))
                            if show_animation:
                                plt.pause(0.01)

                        if not moved:
                            print("No valid move found, retrying...")
                        continue
            else:
                oscillation_count = 0
        # 实时更新动画

            ax.plot(ix, iy, ".y")  # 标记当前位置（红色点）
            if save_gif:
                fig.canvas.draw()
                frames.append(get_frame_as_array(fig))

            if show_animation:
                plt.pause(0.01)
        if save_gif and frames:
            imageio.mimsave(output_path, frames, fps=15, loop=1)
            print(f"GIF动画已保存至: {output_path}")
    print("Goal!!")  # 到达目标点

    return rx, ry


def draw_heatmap(data):
    """绘制势场热图，直观展示势场分布"""
    data = np.array(data).T  # 转置数据以正确显示
    # 绘制热图，vmax设置颜色最大值，cmap设置颜色映射
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)

def generate_obstacles(sx, sy, gx, gy, n_obstacles, margin=1.0):
    """
    在起点和终点中间区域随机生成障碍物
    参数:
        sx, sy: 起点
        gx, gy: 终点
        n_obstacles: 障碍物数量
        margin: 与边界的最小间距
    返回:
        ox, oy: 障碍物坐标列表
    """
    ox, oy = [], []
    for _ in range(n_obstacles):
        x = random.uniform(min(sx, gx) + margin, max(sx, gx) - margin)
        y = random.uniform(min(sy, gy) + margin, max(sy, gy) - margin)
        ox.append(x)
        oy.append(y)
    return ox, oy


def main():
    """主函数：设置参数并执行路径规划"""
    print("potential_field_planning start")

    # 路径规划参数设置
    sx = 0.0  # 起点x坐标 [米]
    sy = 10.0  # 起点y坐标 [米]
    gx = 30.0  # 目标点x坐标 [米]
    gy = 30.0  # 目标点y坐标 [米]
    grid_size = 0.5  # 势场网格分辨率 [米]
    robot_radius = 5.0  # 机器人半径 [米]

    # 障碍物坐标列表
    #ox = [15.0, 5.0, 20.0, 25.0]  # 障碍物x坐标 [米]
    #oy = [25.0, 15.0, 26.0, 25.0]  # 障碍物y坐标 [米]
    n_obstacles = 30  # 控制障碍物数量
    ox, oy = generate_obstacles(sx, sy, gx, gy, n_obstacles)

    # 生成路径
    _, _ = potential_field_planning(
        sx, sy, gx, gy, ox, oy, grid_size, robot_radius)

    # 显示最终动画

if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
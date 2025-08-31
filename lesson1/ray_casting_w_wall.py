import numpy as np
import matplotlib.pyplot as plt
import math
import imageio
from utils.plot import get_frame_as_array


# 参数配置
WALL_THICKNESS = 0.3  # 墙体厚度，单位：米
DOOR_WIDTH = 0.8  # 门的宽度，单位：米
GRID_RESOLUTION = 0.1  # 栅格分辨率
ANGULAR_RESOLUTION = 1  # 角度分辨率（度）
SAFETY_MARGIN = 0.02  # 安全余量，用于避免浮点数精度误差

show_animation = True
save_gif = True
output_path = "ray_casting_improved.gif"

# 墙体信息 - [起点x, 起点y, 终点x, 终点y, 门位置(None表示无门)]
walls = [
    [0, 0, 10, 0, None],
    [0, 8, 10, 8, None],
    [0, 0, 0, 8, None],
    [10, 0, 10, 8, None],
    [2, 0, 2, 6, 3],
    [6, 0, 6, 6, 3],
    [2, 2, 6, 2, 4],
    [2, 6, 6, 6, 4],
    [6, 2, 6, 4, 3],
    [6, 4, 8, 4, None],
    [8, 2, 8, 4, 3],
    [6, 2, 8, 2, None],
    [0, 2, 2, 2, 1],
    [6, 2, 10, 2, 8.5],
    [0, 6, 2, 6, 1],
    [6, 6, 10, 6, 8]
]

# 房间信息 - [房间名称: (传感器位置x, 传感器位置y)]
rooms = {
    "Room 1": (1, 1), "Room 2": (4, 1), "Room 3": (7, 1),
    "Room 4": (1, 4), "Room 5": (4, 4), "Room 6": (7, 3),
    "Room 7": (8, 5), "Room 8": (4, 7)
}


def set_wall_grid(x_min, x_max, y_min, y_max, grid_res, pmap, minx, miny, value=1.0):
    """设置矩形范围内的所有栅格值"""
    # 转换为栅格索引范围
    ix_min = int(np.floor((x_min - minx) / grid_res))
    ix_max = int(np.ceil((x_max - minx) / grid_res))
    iy_min = int(np.floor((y_min - miny) / grid_res))
    iy_max = int(np.ceil((y_max - miny) / grid_res))

    # 遍历并设置所有栅格
    for ix in range(ix_min, ix_max + 1):
        for iy in range(iy_min, iy_max + 1):
            if 0 <= ix < len(pmap) and 0 <= iy < len(pmap[0]):
                pmap[ix][iy] = value  # 设置栅格值


def extract_obstacles(walls, xyreso, minx, miny, xw, yw):
    pmap = [[0.0 for _ in range(yw)] for _ in range(xw)]

    for wall in walls:
        x1, y1, x2, y2, door_pos = wall

        # 计算墙体方向和法向量
        if x1 == x2:  # 垂直墙体
            center_x = x1
            x_min = center_x - WALL_THICKNESS / 2
            x_max = center_x + WALL_THICKNESS / 2
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            # 标记墙体区域为占据
            set_wall_grid(x_min, x_max, y_min, y_max, xyreso, pmap, minx, miny, 1.0)

            if door_pos is not None:
                door_start = door_pos - DOOR_WIDTH / 2
                door_end = door_pos + DOOR_WIDTH / 2
                # 清除门区域（设为0.0）
                set_wall_grid(x_min, x_max, door_start, door_end, xyreso, pmap, minx, miny, 0.0)

        else:  # 水平墙体
            center_y = y1
            y_min = center_y - WALL_THICKNESS / 2
            y_max = center_y + WALL_THICKNESS / 2
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            # 标记墙体区域为占据
            set_wall_grid(x_min, x_max, y_min, y_max, xyreso, pmap, minx, miny, 1.0)

            if door_pos is not None:
                door_start = door_pos - DOOR_WIDTH / 2
                door_end = door_pos + DOOR_WIDTH / 2
                # 清除门区域（设为0.0）
                set_wall_grid(door_start, door_end, y_min, y_max, xyreso, pmap, minx, miny, 0.0)

    ox, oy = [], []
    for ix in range(len(pmap)):
        for iy in range(len(pmap[0])):
            if pmap[ix][iy] == 1.0:
                # 计算栅格中心的实际坐标
                x = ix * xyreso + minx + xyreso / 2
                y = iy * xyreso + miny + xyreso / 2
                ox.append(x)
                oy.append(y)

    return ox, oy, pmap


def calc_grid_map_config(ox, oy, xyreso):
    """计算栅格地图配置"""
    minx = np.floor(min(ox)) - 1
    miny = np.floor(min(oy)) - 1
    maxx = np.ceil(max(ox)) + 1
    maxy = np.ceil(max(oy)) + 1

    # 限制最大地图尺寸，避免内存溢出
    max_grid_size = 1000
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))

    if xw > max_grid_size or yw > max_grid_size:
        scale = max(xw, yw) / max_grid_size
        new_xyreso = xyreso * scale
        print(f"警告: 栅格尺寸过大 ({xw}x{yw})，自动调整分辨率为 {new_xyreso:.3f}")
        xw = int(round((maxx - minx) / new_xyreso))
        yw = int(round((maxy - miny) / new_xyreso))
        return minx, miny, maxx, maxy, xw, yw, new_xyreso
    else:
        return minx, miny, maxx, maxy, xw, yw, xyreso


class precastDB:
    def __init__(self):
        self.px, self.py, self.d = 0.0, 0.0, 0.0
        self.angle, self.ix, self.iy = 0.0, 0, 0


def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    return angle if angle >= 0 else angle + 2 * math.pi


def pre_casting(minx, miny, xw, yw, xyreso, yawreso):
    """预计算射线投射数据"""
    precast = [[] for _ in range(int(round((math.pi * 2) / yawreso)) + 1)]
    max_distance = 15.0  # 最大预计算距离

    for ix in range(xw):
        for iy in range(yw):
            px = ix * xyreso + minx
            py = iy * xyreso + miny

            if math.hypot(px, py) > max_distance:
                continue

            d = math.hypot(px, py)
            angle = atan_zero_to_twopi(py, px)
            angleid = int(math.floor(angle / yawreso))
            pc = precastDB()
            pc.px, pc.py, pc.d = px, py, d
            pc.ix, pc.iy, pc.angle = ix, iy, angle
            precast[angleid].append(pc)
    return precast


def generate_ray_casting_grid_map(ox, oy, xyreso, yawreso, sensor_x=0, sensor_y=0):
    """生成栅格地图，增强边缘处理"""
    # 调整障碍物坐标到传感器坐标系
    ox_rel = [x - sensor_x for x in ox]
    oy_rel = [y - sensor_y for y in oy]

    # 计算栅格地图配置
    minx, miny, maxx, maxy, xw, yw, actual_xyreso = calc_grid_map_config(ox_rel, oy_rel, xyreso)

    # 创建栅格地图
    pmap = [[0.0 for _ in range(yw)] for _ in range(xw)]

    # 预计算射线投射数据
    precast = pre_casting(minx, miny, xw, yw, actual_xyreso, yawreso)

    # 处理每个障碍物点
    for (x, y) in zip(ox_rel, oy_rel):
        d = math.hypot(x, y)
        angle = atan_zero_to_twopi(y, x)
        angleid = int(math.floor(angle / yawreso))
        angleid = max(0, min(angleid, len(precast) - 1))
        gridlist = precast[angleid]

        # 确保索引有效
        ix = int(round((x - minx) / actual_xyreso))
        iy = int(round((y - miny) / actual_xyreso))
        ix = np.clip(ix, 0, xw - 1)
        iy = np.clip(iy, 0, yw - 1)

        # 更新栅格地图
        for grid in gridlist:
            gx = np.clip(grid.ix, 0, xw - 1)
            gy = np.clip(grid.iy, 0, yw - 1)
            if grid.d > d:
                pmap[gx][gy] = 0.5  # 可见但未被占据
        pmap[ix][iy] = 1.0  # 障碍物位置

    # 特殊处理：确保墙体边缘完全阻挡光线
    for wall in walls:
        # 计算墙体在传感器坐标系中的相对位置
        x1_rel, y1_rel = wall[0] - sensor_x, wall[1] - sensor_y
        x2_rel, y2_rel = wall[2] - sensor_x, wall[3] - sensor_y

        # 检查墙体是否在当前传感器的可见范围内
        if not is_wall_in_range(x1_rel, y1_rel, x2_rel, y2_rel, minx, maxx, miny, maxy):
            continue

        # 特殊处理墙体边缘的栅格
        enhance_wall_edges(pmap, wall, sensor_x, sensor_y, minx, miny, actual_xyreso)

    return pmap, minx + sensor_x, maxx + sensor_x, miny + sensor_y, maxy + sensor_y, actual_xyreso


def is_wall_in_range(x1, y1, x2, y2, minx, maxx, miny, maxy):
    """检查墙体是否在指定范围内"""
    # 简单扩展边界以考虑墙体厚度
    expand = WALL_THICKNESS + 0.5
    return (max(x1, x2) >= minx - expand and min(x1, x2) <= maxx + expand and
            max(y1, y2) >= miny - expand and min(y1, y2) <= maxy + expand)


def enhance_wall_edges(pmap, wall, sensor_x, sensor_y, minx, miny, xyreso):
    """增强墙体边缘的栅格，确保完全阻挡光线"""
    x1, y1, x2, y2, door_pos = wall

    # 计算墙体方向
    if x1 == x2:  # 垂直墙体
        # 沿墙体方向采样
        y_samples = np.linspace(y1, y2, max(int((y2 - y1) / GRID_RESOLUTION), 5))

        # 墙体边缘的厚度方向采样
        for y in y_samples:
            # 检查是否在门的区域
            if door_pos is not None:
                door_start = door_pos - DOOR_WIDTH / 2
                door_end = door_pos + DOOR_WIDTH / 2
                if door_start <= y <= door_end:
                    continue

            # 墙体两侧边缘
            for edge_x in [x1 - WALL_THICKNESS / 2, x1 + WALL_THICKNESS / 2]:
                # 转换到栅格坐标
                ix = int(round((edge_x - sensor_x - minx) / xyreso))
                iy = int(round((y - sensor_y - miny) / xyreso))

                # 确保索引有效
                if 0 <= ix < len(pmap) and 0 <= iy < len(pmap[0]):
                    pmap[ix][iy] = 1.0  # 确保边缘栅格被标记为障碍物

    else:  # 水平墙体
        # 沿墙体方向采样
        x_samples = np.linspace(x1, x2, max(int((x2 - x1) / GRID_RESOLUTION), 5))

        # 墙体边缘的厚度方向采样
        for x in x_samples:
            # 检查是否在门的区域
            if door_pos is not None:
                door_start = door_pos - DOOR_WIDTH / 2
                door_end = door_pos + DOOR_WIDTH / 2
                if door_start <= x <= door_end:
                    continue

            # 墙体两侧边缘
            for edge_y in [y1 - WALL_THICKNESS / 2, y1 + WALL_THICKNESS / 2]:
                # 转换到栅格坐标
                ix = int(round((x - sensor_x - minx) / xyreso))
                iy = int(round((edge_y - sensor_y - miny) / xyreso))

                # 确保索引有效
                if 0 <= ix < len(pmap) and 0 <= iy < len(pmap[0]):
                    pmap[ix][iy] = 1.0  # 确保边缘栅格被标记为障碍物


def draw_heatmap(data, minx, maxx, miny, maxy, xyreso, ax=None):
    if ax is None:
        ax = plt.gca()
    data = np.array(data).T
    rows, cols = data.shape
    x = np.linspace(minx - xyreso / 2, maxx + xyreso / 2, cols + 1)
    y = np.linspace(miny - xyreso / 2, maxy + xyreso / 2, rows + 1)
    X, Y = np.meshgrid(x, y)
    im = ax.pcolor(X, Y, data, vmax=1.0, cmap=plt.cm.Blues)
    return im


def main():
    print("改进版光线投射栅格地图生成（整合mark_wall_grid和clear_wall_grid）")

    # 预计算全局栅格地图配置
    temp_ox, temp_oy = [], []
    for wall in walls:
        x1, y1, x2, y2, _ = wall
        temp_ox.extend([x1, x2])
        temp_oy.extend([y1, y2])
    minx, miny, maxx, maxy, xw, yw, actual_xyreso = calc_grid_map_config(temp_ox, temp_oy, GRID_RESOLUTION)

    if save_gif:
        frames = []
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt.subplots(figsize=(10, 10))

    for room in rooms:
        sensor_x, sensor_y = rooms[room]
        # 使用新的extract_obstacles函数生成障碍物点和栅格地图
        ox, oy, pmap = extract_obstacles(walls, actual_xyreso, minx, miny, xw, yw)

        ax.cla()
        draw_heatmap(pmap, minx, maxx, miny, maxy, actual_xyreso, ax)

        if show_animation or save_gif:
            # 绘制墙体和门
            for wall in walls:
                x1, y1, x2, y2, door_pos = wall
                if door_pos is None:
                    # 无门墙体
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=WALL_THICKNESS * 10)
                else:
                    if x1 == x2:  # 垂直墙体
                        # 绘制门上方和下方的墙体
                        if y1 < door_pos - DOOR_WIDTH / 2:
                            ax.plot([x1, x1], [y1, door_pos - DOOR_WIDTH / 2], 'k-', linewidth=WALL_THICKNESS * 10)
                        if y2 > door_pos + DOOR_WIDTH / 2:
                            ax.plot([x1, x1], [door_pos + DOOR_WIDTH / 2, y2], 'k-', linewidth=WALL_THICKNESS * 10)
                    else:  # 水平墙体
                        # 绘制门左侧和右侧的墙体
                        if x1 < door_pos - DOOR_WIDTH / 2:
                            ax.plot([x1, door_pos - DOOR_WIDTH / 2], [y1, y1], 'k-', linewidth=WALL_THICKNESS * 10)
                        if x2 > door_pos + DOOR_WIDTH / 2:
                            ax.plot([door_pos + DOOR_WIDTH / 2, x2], [y1, y1], 'k-', linewidth=WALL_THICKNESS * 10)

            # 绘制传感器位置
            ax.plot(sensor_x, sensor_y, "ob", markersize=12, label=f"sensor at: {room}")

            # 绘制房间标签
            for name, (x, y) in rooms.items():
                ax.text(x, y, name, fontsize=9, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7))

            ax.set_title("")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-0.5, 8.5)
            ax.set_aspect('equal')
            ax.legend(loc="upper right")
            plt.tight_layout()

            if show_animation:
                plt.draw()
                plt.pause(1.0)
            if save_gif:
                frame = get_frame_as_array(fig)
                frames.append(frame)

    if save_gif:
        imageio.mimsave(output_path, frames, fps=1, loop=0)
        print(f"GIF已保存至 {output_path}")

    if show_animation and not save_gif:
        plt.show()


if __name__ == '__main__':
    main()
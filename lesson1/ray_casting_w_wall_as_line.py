import math
import numpy as np
import matplotlib.pyplot as plt
import imageio
from utils.plot import get_frame_as_array


# 墙体参数
DOOR_WIDTH = 0.8
MAX_DISTANCE = 10
show_animation = True
save_gif = True
output_path = "ray_casting_fixed.gif"

# 墙体信息（(x, y)坐标）
walls = [
    {"start": (0, 0), "end": (10, 0), "door_pos": None},    # 底部外墙
    {"start": (0, 10), "end": (10, 10), "door_pos": None},    # 顶部外墙
    {"start": (0, 0), "end": (0, 10), "door_pos": None},     # 左侧外墙
    {"start": (10, 0), "end": (10, 10), "door_pos": None},   # 右侧外墙
    {"start": (2, 0), "end": (2, 6), "door_pos": 3},
    {"start": (6, 0), "end": (6, 6), "door_pos": 3},
    {"start": (2, 2), "end": (6, 2), "door_pos": 4},
    {"start": (2, 6), "end": (6, 6), "door_pos": 4},
    {"start": (6, 2), "end": (6, 4), "door_pos": 3},
    {"start": (6, 4), "end": (8, 4), "door_pos": None},
    {"start": (8, 2), "end": (8, 4), "door_pos": 3},
    {"start": (6, 2), "end": (8, 2), "door_pos": None},
    {"start": (0, 2), "end": (2, 2), "door_pos": 1},
    {"start": (6, 2), "end": (10, 2), "door_pos": 8.5},
    {"start": (0, 6), "end": (2, 6), "door_pos": 1},
    {"start": (6, 6), "end": (10, 6), "door_pos": 8}
]

rooms = {"Room 1": (1, 1), "Room 2": (4, 1), "Room 3": (7, 1),
    "Room 4": (1, 4), "Room 5": (4, 4), "Room 6": (7, 3),
    "Room 7": (8, 5), "Room 8": (4, 7)}

def point_on_segment(px, py, x1, y1, x2, y2):
    """更精确地判断点是否在线段上"""
    # 缩小范围判断，避免误判
    if (min(x1, x2) - 1e-3 <= px <= max(x1, x2) + 1e-3 and
        min(y1, y2) - 1e-3 <= py <= max(y1, y2) + 1e-3):
        # 降低叉积阈值，提高精度
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if abs(cross) < 1e-6:
            return True
    return False

# @ 射线起点； @射线方向  @线段起点 @线段终点
def line_segment_intersection(ray_start, ray_dir, segment_start, segment_end):
    """修正射线与线段交点计算，严格判断线段范围"""
    d = np.array(ray_dir, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return None, None
    d /= norm
    p0 = np.array(ray_start, dtype=float)
    a = np.array(segment_start, dtype=float)
    b = np.array(segment_end, dtype=float)
    v = b - a
    denom = d[0] * v[1] - d[1] * v[0]
    if abs(denom) < 1e-12: # 法向相乘为0，即平行
        return None, None  # 平行或共线
    w = a - p0
    t = (w[0] * v[1] - w[1] * v[0]) / denom
    u = (w[0] * d[1] - w[1] * d[0]) / denom

    # 严格限制：射线向前（t≥0）且交点在线段上（0≤u≤1）
    if t >= 0 and 0 <= u <= 1:
        pt = p0 + t * d
        return t, tuple(pt)
    return None, None


def get_wall_segments(walls):
    """修正墙体线段生成，确保门的分割正确"""
    segments = []
    for wall in walls:
        x1, y1 = wall["start"]
        x2, y2 = wall["end"]
        door_pos = wall["door_pos"]

        if door_pos is None:
            segments.append({"start": (x1, y1), "end": (x2, y2)})
            continue

        # 处理垂直墙体（x固定）
        if math.isclose(x1, x2):
            y_min, y_max = min(y1, y2), max(y1, y2)
            y_top = door_pos - DOOR_WIDTH / 2  # 门的上沿
            y_bottom = door_pos + DOOR_WIDTH / 2  # 门的下沿
            # 添加上方墙体（从y_min到门的上沿）
            if y_min < y_top:
                segments.append({"start": (x1, y_min), "end": (x1, y_top)})
            # 添加下方墙体（从门的下沿到y_max）
            if y_bottom < y_max:
                segments.append({"start": (x1, y_bottom), "end": (x1, y_max)})
        # 处理水平墙体（y固定）
        else:
            y = y1
            # 确保x1 < x2，统一处理逻辑
            x_min, x_max = sorted((x1, x2))
            x_left = door_pos - DOOR_WIDTH / 2  # 门的左沿
            x_right = door_pos + DOOR_WIDTH / 2  # 门的右沿
            # 添加左侧墙体（从x_min到门的左沿）
            if x_min < x_left:
                segments.append({"start": (x_min, y), "end": (x_left, y)})
            # 添加右侧墙体（从门的右沿到x_max）
            if x_right < x_max:
                segments.append({"start": (x_right, y), "end": (x_max, y)})
    return segments


def cast_ray(sensor_pos, angle, wall_segments, max_dist):
    """修正射线投射：无交点时返回最大距离点"""
    ray_dir = (np.cos(angle), np.sin(angle))  # 单位方向向量
    min_t = max_dist  # 最近交点的距离
    closest_pt = None

    for seg in wall_segments:
        t, pt = line_segment_intersection(sensor_pos, ray_dir, seg['start'], seg['end'])
        # 记录最近的有效交点（距离≤max_dist）
        if t is not None and t <= min_t:
            min_t = t
            closest_pt = pt

    # 无交点时，返回最大距离处的点
    if closest_pt is None:
        return (sensor_pos[0] + max_dist * ray_dir[0], sensor_pos[1] + max_dist * ray_dir[1])
    return closest_pt


def generate_ray_casting_map(sensor_pos, wall_segments, num_rays=360):
    """确保所有方向都有光线（包括无墙方向）"""
    # 检查并调整传感器位置（避免在墙内）
    for seg in wall_segments:
        x1, y1 = seg["start"]
        x2, y2 = seg["end"]
        if point_on_segment(sensor_pos[0], sensor_pos[1], x1, y1, x2, y2):
            print(f"警告：传感器位于墙内，调整位置：{sensor_pos}")
            # 向斜下方微调（避免再次进入墙内）
            sensor_pos = (sensor_pos[0] + 0.1, sensor_pos[1] - 0.1)

    angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    points = []
    for angle in angles:
        # print(angle)
        # 每个方向都必须有一个点（墙体或最大距离）
        pt = cast_ray(sensor_pos, angle, wall_segments, MAX_DISTANCE)
        points.append(pt)
    return points


def draw_ray_casting_result(sensor_pos, points, wall_segments, ax=None):
    if ax is None:
        ax = plt.gca()

    # 绘制墙体
    for seg in wall_segments:
        x1, y1 = seg["start"]
        x2, y2 = seg["end"]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=5)

    # 绘制传感器
    ax.plot(sensor_pos[0], sensor_pos[1], "bo", markersize=10, label="Sensor")

    # 绘制射线和交点
    for pt in points:
        ax.plot([sensor_pos[0], pt[0]], [sensor_pos[1], pt[1]], 'b-', alpha=0.2)
        ax.plot(pt[0], pt[1], 'ro', markersize=2)

    for name, (x, y) in rooms.items():
        ax.text(x, y, name, fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7))

    # 设置坐标轴
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title(f"Ray Casting - Sensor at {sensor_pos}")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal')
    ax.legend()


def main():
    # 获取墙体线段
    wall_segments = get_wall_segments(walls)
    # 打印有效墙体线段数量
    print(f"有效墙体线段数量：{len(wall_segments)}")

    # 如果需要保存gif，则创建一个空的frames列表和一个figure对象
    if save_gif:
        frames = []
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
    # 否则，直接创建一个figure对象和ax对象
    else:
        fig, ax = plt.subplots(figsize=(10, 8))

    # 遍历rooms字典中的每个房间
    for room_name, sensor_pos in rooms.items():
        # 清空ax对象
        ax.cla()
        # 生成射线投射地图
        points = generate_ray_casting_map(sensor_pos, wall_segments)
        # 绘制射线投射结果
        draw_ray_casting_result(sensor_pos, points, wall_segments, ax)

        # 如果需要显示动画，则调整布局并暂停0.01秒
        if show_animation:
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
        # 如果需要保存gif，则将当前帧添加到frames列表中
        if save_gif:
            frames.append(get_frame_as_array(fig))

    # 如果需要保存gif，则将frames列表保存为gif文件
    if save_gif:
        imageio.mimsave(output_path, frames, fps=2, loop=0)
        print(f"GIF已保存至：{output_path}")

    # 如果不需要保存gif，但需要显示动画，则显示plt对象
    if not save_gif and show_animation:
        plt.show()


if __name__ == '__main__':
    main()
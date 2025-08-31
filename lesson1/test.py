# 完整重构：射线与线段交点检测 + 可视化绘图
import math
import numpy as np
import matplotlib.pyplot as plt

DOOR_WIDTH = 0.8


def get_wall_segments(walls):
    """根据墙体及门位置拆分成线段"""
    segments = []
    for wall in walls:
        x1, y1 = wall['start']
        x2, y2 = wall['end']
        door_pos = wall.get('door_pos')

        # 无门的完整墙体
        if door_pos is None:
            segments.append({'start': (x1, y1), 'end': (x2, y2)})
            continue

        # 垂直墙体（x 相同）
        if math.isclose(x1, x2):
            y_min, y_max = sorted((y1, y2))
            top = door_pos - DOOR_WIDTH / 2
            bottom = door_pos + DOOR_WIDTH / 2
            # 门上方
            if y_min < top:
                segments.append({'start': (x1, y_min), 'end': (x1, top)})
            # 门下方
            if bottom < y_max:
                segments.append({'start': (x1, bottom), 'end': (x1, y_max)})
        # 水平墙体（y 相同）
        else:
            x_min, x_max = sorted((x1, x2))
            left = door_pos - DOOR_WIDTH / 2
            right = door_pos + DOOR_WIDTH / 2
            # 门左侧
            if x_min < left:
                segments.append({'start': (x_min, y1), 'end': (left, y1)})
            # 门右侧
            if right < x_max:
                segments.append({'start': (right, y1), 'end': (x_max, y1)})
    return segments


def line_segment_intersection(ray_start, ray_dir, seg_start, seg_end):
    """计算射线 ray_start + t*ray_dir 与线段 seg_start->seg_end 的交点 t,u，并返回第一个交点"""
    # 归一化方向
    d = np.array(ray_dir, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return None, None
    d /= norm

    p0 = np.array(ray_start, dtype=float)
    a = np.array(seg_start, dtype=float)
    b = np.array(seg_end, dtype=float)

    v = b - a
    denom = d[0] * v[1] - d[1] * v[0]
    # 平行或共线
    if abs(denom) < 1e-10:
        return None, None

    w = a - p0
    t = (w[0] * v[1] - w[1] * v[0]) / denom
    u = (w[0] * d[1] - w[1] * d[0]) / denom

    # 射线前方 t>=0，线段内 u∈[0,1]
    if t >= 0 and 0 <= u <= 1:
        pt = p0 + t * d
        return t, tuple(pt)
    return None, None


def cast_ray(sensor_pos, angle, wall_segments, max_dist):
    """在指定方向上投射一条射线，返回与最近墙体的交点或最大距离点"""
    dir_vec = (math.cos(angle), math.sin(angle))
    closest_pt = None
    min_t = max_dist

    for seg in wall_segments:
        t, pt = line_segment_intersection(sensor_pos, dir_vec, seg['start'], seg['end'])
        if t is not None and t < min_t:
            min_t = t
            closest_pt = pt

    if closest_pt is None:
        # 未相交，返回最大距离点
        return (sensor_pos[0] + max_dist * dir_vec[0], sensor_pos[1] + max_dist * dir_vec[1])
    return closest_pt


def visualize(sensor_pos, wall_segments, angles, hits, max_dist):
    """绘制墙体、传感器、射线及交点"""
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')

    # 绘墙体
    for seg in wall_segments:
        x0,y0 = seg['start']; x1,y1 = seg['end']
        ax.plot([x0,x1],[y0,y1],'k-',linewidth=2)
    # 传感器
    ax.plot(sensor_pos[0],sensor_pos[1],'bs',markersize=8,label='Sensor')

    # 射线
    for ang, pt in zip(angles, hits):
        ax.plot([sensor_pos[0],pt[0]],[sensor_pos[1],pt[1]],'r--',alpha=0.6)
        ax.plot(pt[0],pt[1],'ro',markersize=4)

    # 最大距离圆
    theta = np.linspace(0,2*math.pi,200)
    circ = np.stack([sensor_pos[0] + max_dist*np.cos(theta),sensor_pos[1]+max_dist*np.sin(theta)])
    ax.plot(circ[0],circ[1],'g:',alpha=0.5,label=f'max_dist={max_dist}')

    ax.legend(); ax.grid(True)
    ax.set_xlim(-1,6); ax.set_ylim(-1,6)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('Ray Casting Visualization')
    plt.show()


# 示例运行
if __name__ == '__main__':
    walls = [
        {'start':(0,0),'end':(5,0),'door_pos':None},
        {'start':(0,5),'end':(5,5),'door_pos':None},
        {'start':(0,0),'end':(0,5),'door_pos':None},
        {'start':(5,0),'end':(5,5),'door_pos':None},
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
    sensor = (1,1)
    max_d = 8
    angles = np.linspace(0,math.pi*2,360,endpoint=False)
    segs = get_wall_segments(walls)
    hits = [cast_ray(sensor,a,segs,max_d) for a in angles]
    visualize(sensor,segs,angles,hits,max_d)

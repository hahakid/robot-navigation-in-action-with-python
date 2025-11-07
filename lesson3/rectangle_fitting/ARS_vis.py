import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------ 生成聚集点 ------------------
np.random.seed(42)
num_clusters = 4
points_per_cluster = 8
cluster_centers = np.random.uniform(-10, 10, (num_clusters, 2))

ox, oy = [], []
for cx, cy in cluster_centers:
    ox.extend(cx + np.random.normal(0, 1.5, points_per_cluster))
    oy.extend(cy + np.random.normal(0, 1.5, points_per_cluster))
ox = np.array(ox)
oy = np.array(oy)

R0 = 3.0
Rd = 0.01


# ------------------ 簇合并逻辑 ------------------
def adoptive_range_segmentation_steps(ox, oy, R0=R0, Rd=Rd):
    segment_list = []
    for i, _ in enumerate(ox):
        c = set()
        r = R0 + Rd * np.linalg.norm([ox[i], oy[i]])
        for j, _ in enumerate(ox):
            d = np.hypot(ox[i] - ox[j], oy[i] - oy[j])
            if d <= r:
                c.add(j)
        segment_list.append(c)

    # 记录每一步的簇状态
    steps = [segment_list.copy()]

    while True:
        no_change = True
        for (c1, c2) in list(itertools.permutations(range(len(segment_list)), 2)):
            if segment_list[c1] & segment_list[c2]:
                segment_list[c1] = (segment_list[c1] | segment_list.pop(c2))
                no_change = False
                steps.append(segment_list.copy())
                break
        if no_change:
            break
    return steps


# ------------------ 绘图函数 ------------------
def plot_clusters(ax, ox, oy, segments):
    ax.clear()
    colors = cm.get_cmap('tab20', len(segments))
    for idx, s in enumerate(segments):
        pts_x = [ox[i] for i in s]
        pts_y = [oy[i] for i in s]
        ax.scatter(pts_x, pts_y, color=colors(idx), s=100)
        center_x = np.mean(pts_x)
        center_y = np.mean(pts_y)
        ax.text(center_x, center_y, str(idx), fontsize=12, ha='center', va='center', weight='bold')
    ax.set_xlim(ox.min() - 2, ox.max() + 2)
    ax.set_ylim(oy.min() - 2, oy.max() + 2)
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    # 图注放在坐标系外
    ax.legend([f"Cluster {i}" for i in range(len(segments))],
              bbox_to_anchor=(1.05, 1), loc='upper left')


# ------------------ 动态可视化 ------------------
steps = adoptive_range_segmentation_steps(ox, oy)

fig, ax = plt.subplots(figsize=(7, 7))
fig.subplots_adjust(left=0.05, right=0.75, top=0.95, bottom=0.1)

def update(frame):
    plot_clusters(ax, ox, oy, steps[frame])
    ax.set_title(f"Step {frame}: {len(steps[frame])} clusters")


ani = FuncAnimation(fig, update, frames=len(steps), interval=1000, repeat=False)

# ------------------ 保存 GIF ------------------
ani.save("cluster_merge.gif", writer=PillowWriter(fps=1))


import numpy as np
import matplotlib.pyplot as plt
import cubic_spline_planner
from pure_pursuit import plot_vehicle

# ==== 固定参数 ====
L = 2.9  # 轴距
class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x  # x坐标
        self.y = y  # y坐标
        self.yaw = yaw  # 偏航角（航向角）
        self.v = v  # 行驶速度

ax = [0.0, 100.0, 100.0, 50.0, 60.0]  # x方向控制点
ay = [0.0, 0.0, -30.0, -20.0, 0.0]   # y方向控制点

cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)

state = State(x=-0.0, y=5.0, yaw=np.radians(20.0), v=0.0) # 车辆在原点，航向30°

# ==== 计算前轴位置 ====
fx = state.x + L * np.cos(state.yaw)
fy = state.y + L * np.sin(state.yaw)

# ==== 找最近路径点 ====
dx = [fx - icx for icx in cx]
dy = [fy - icy for icy in cy]
d = np.hypot(dx, dy)
target_idx = np.argmin(d)

# ==== 计算横向误差 ====
front_axle_vec = [-np.cos(state.yaw + np.pi / 2), -np.sin(state.yaw + np.pi / 2)]
error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

# ==== 可视化 ====
plt.figure(figsize=(8, 8))

plt.plot(cx, cy, ".r", label="cx-cy")
# plt.plot(cx, cy, "k-", label="reference path")                # 路径
plt.plot(state.x, state.y, "bo", label="x-y", markersize=3)       # 车辆
plt.plot(fx, fy, "go", label="fx-fy")               # 前轴
plt.plot(cx[target_idx], cy[target_idx], "k*", label="1st target point", markersize=8)  # 目标点
plot_vehicle(state.x, state.y, state.yaw, -cyaw[0], is_reverse=False)
# 车辆朝向箭头
plt.arrow(state.x, state.y, np.cos(state.yaw), np.sin(state.yaw),
          head_width=0.3, color="b", length_includes_head=True)

# 车辆 -> 前轴的线
plt.plot([state.x, fx], [state.y, fy], "g--", label="vc-2-fxy")

# 前轴 -> 目标点的线（横向误差线）
plt.plot([fx, cx[target_idx]], [fy, cy[target_idx]], "r--", label="fxy-2-target-1st")

# 横向误差方向箭头（在前轴处画）
plt.arrow(fx, fy, front_axle_vec[0], front_axle_vec[1],
          head_width=0.2, color="m", length_includes_head=True, label="fxy-2-")

plt.xlim((-10, 10))
plt.ylim((-10, 10))

plt.axis("equal")
plt.legend()
plt.title(f"target idx={target_idx}, fae={error_front_axle:.2f}")
plt.show()

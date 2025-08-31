import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号乱码
# 权重示例
weights = np.array([0.1, 0.2, 0.6, 0.1])
cum_weights = np.cumsum(weights)

# 生成等间隔的 resample_id 指标点（带小随机抖动）
N = len(weights)
base = np.arange(N) / N
resample_id = base + np.random.rand(N) / N

# 绘图
fig, ax = plt.subplots(figsize=(10, 2))

# 绘制CDF线段
ax.hlines(1, 0, 1, color="lightgray", linestyle="--")
ax.hlines(0, 0, 1, color="lightgray", linestyle="--")

# 绘制权重区间
prev = 0
colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
for i, w in enumerate(weights):
    ax.plot([prev, cum_weights[i]], [0.5, 0.5], color=colors[i], linewidth=10, solid_capstyle="butt")
    ax.text((prev + cum_weights[i]) / 2, 0.7, f"粒子{i+1}\n({w:.1f})", ha="center", fontsize=10)
    prev = cum_weights[i]

# 绘制 resample_id 指标点
ax.scatter(resample_id, [0.5]*N, color="black", zorder=5)
for i, r in enumerate(resample_id):
    ax.text(r, 0.35, f"{i+1}", ha="center", fontsize=9, color="black")

# 细节调整
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("CDF (累积分布函数)")
ax.set_title("低方差重采样示意图：指标点落入不同权重区间")

plt.show()

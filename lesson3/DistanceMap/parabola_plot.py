import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# 定义抛物线函数
def parabola(x, q, dq=0):
    """计算抛物线值: f(x) = (x - q)^2 + dq"""
    return (x - q) ** 2 + dq


# 计算两个抛物线的交点
def find_intersection(q1, d1, q2, d2):
    """计算两个抛物线的交点横坐标"""
    if q1 == q2:
        return float('inf') if d1 > d2 else -float('inf')
    numerator = (d2 + q2 ** 2) - (d1 + q1 ** 2)
    denominator = 2 * (q2 - q1)
    return numerator / denominator


# 绘制基本抛物线及其交点
def plot_basic_parabolas():
    x = np.linspace(-5, 15, 1000)

    # 两个抛物线参数
    q1, d1 = 2, 0  # 第一个障碍物点
    q2, d2 = 8, 0  # 第二个障碍物点

    # 计算交点
    s = find_intersection(q1, d1, q2, d2)

    # 绘制抛物线
    plt.figure(figsize=(10, 6))
    plt.plot(x, parabola(x, q1, d1), 'b-', label=f'抛物线 q={q1}')
    plt.plot(x, parabola(x, q2, d2), 'r-', label=f'抛物线 q={q2}')

    # 绘制交点
    plt.plot(s, parabola(s, q1, d1), 'go', label=f'交点 s={s:.2f}')
    plt.axvline(x=s, color='g', linestyle='--', alpha=0.5)

    # 绘制包络线
    envelope = np.minimum(parabola(x, q1, d1), parabola(x, q2, d2))
    plt.plot(x, envelope, 'k-', linewidth=2, label='下包络线')

    plt.title('基本抛物线及其下包络线')
    plt.xlabel('x')
    plt.ylabel('距离值')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-1, 30)
    plt.show()


# 绘制新抛物线加入时的两种情况
def plot_addition_scenarios():
    x = np.linspace(-5, 20, 1000)

    # 已有抛物线参数
    q_prev, d_prev = 3, 0  # v[k-1]
    q_curr, d_curr = 10, 0  # v[k]
    z_k = find_intersection(q_prev, d_prev, q_curr, d_curr)  # z[k]

    # 新抛物线参数 (两种情况)
    q_new1, d_new1 = 15, 0  # 情况1: s > z[k]
    q_new2, d_new2 = 7, 0  # 情况2: s <= z[k]

    # 计算交点
    s1 = find_intersection(q_curr, d_curr, q_new1, d_new1)
    s2 = find_intersection(q_curr, d_curr, q_new2, d_new2)

    # 创建子图
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 2, figure=fig)

    # 情况1: s > z[k]
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, parabola(x, q_prev, d_prev), 'b-', label=f'v[k-1] = {q_prev}')
    ax1.plot(x, parabola(x, q_curr, d_curr), 'r-', label=f'v[k] = {q_curr}')
    ax1.plot(x, parabola(x, q_new1, d_new1), 'g-', label=f'新抛物线 q={q_new1}')

    # 绘制交点和辅助线
    ax1.plot(z_k, parabola(z_k, q_prev, d_prev), 'ko', label=f'z[k] = {z_k:.2f}')
    ax1.plot(s1, parabola(s1, q_curr, d_curr), 'mo', label=f's = {s1:.2f}')
    ax1.axvline(x=z_k, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=s1, color='m', linestyle='--', alpha=0.5)

    # 绘制包络线
    envelope1 = np.minimum(np.minimum(parabola(x, q_prev, d_prev),
                                      parabola(x, q_curr, d_curr)),
                           parabola(x, q_new1, d_new1))
    ax1.plot(x, envelope1, 'k-', linewidth=2, label='新下包络线')

    ax1.set_title('情况1: s > z[k] - 新抛物线直接加入')
    ax1.set_xlabel('x')
    ax1.set_ylabel('距离值')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-1, 40)

    # 情况2: s <= z[k]
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, parabola(x, q_prev, d_prev), 'b-', label=f'v[k-1] = {q_prev}')
    ax2.plot(x, parabola(x, q_curr, d_curr), 'r-', label=f'v[k] = {q_curr} (将被移除)')
    ax2.plot(x, parabola(x, q_new2, d_new2), 'g-', label=f'新抛物线 q={q_new2}')

    # 绘制交点和辅助线
    ax2.plot(z_k, parabola(z_k, q_prev, d_prev), 'ko', label=f'z[k] = {z_k:.2f}')
    ax2.plot(s2, parabola(s2, q_curr, d_curr), 'mo', label=f's = {s2:.2f}')
    ax2.axvline(x=z_k, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=s2, color='m', linestyle='--', alpha=0.5)

    # 计算移除v[k]后的新交点和包络线
    s_new = find_intersection(q_prev, d_prev, q_new2, d_new2)
    ax2.plot(s_new, parabola(s_new, q_prev, d_prev), 'co', label=f'新交点 = {s_new:.2f}')
    ax2.axvline(x=s_new, color='c', linestyle='--', alpha=0.5)

    # 绘制包络线
    envelope2 = np.minimum(parabola(x, q_prev, d_prev), parabola(x, q_new2, d_new2))
    ax2.plot(x, envelope2, 'k-', linewidth=2, label='新下包络线')

    ax2.set_title('情况2: s <= z[k] - 需要移除v[k]')
    ax2.set_xlabel('x')
    ax2.set_ylabel('距离值')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-1, 40)

    plt.tight_layout()
    plt.show()


# 绘制完整的包络线形成过程
def plot_envelope_formation():
    # 障碍物点集合
    obstacles = [2, 5, 9, 14]
    x = np.linspace(0, 18, 1000)

    # 创建子图
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)

    # 逐步添加抛物线并展示包络线变化
    for i in range(1, 5):
        ax = fig.add_subplot(gs[(i - 1) // 2, (i - 1) % 2])

        # 绘制已添加的抛物线
        current_obstacles = obstacles[:i]
        for q in current_obstacles:
            ax.plot(x, parabola(x, q), label=f'q={q}')

        # 计算并绘制当前包络线
        envelope = np.full_like(x, float('inf'))
        for q in current_obstacles:
            envelope = np.minimum(envelope, parabola(x, q))
        ax.plot(x, envelope, 'k-', linewidth=2, label='下包络线')

        # 标记交点
        for j in range(len(current_obstacles) - 1):
            q1, q2 = current_obstacles[j], current_obstacles[j + 1]
            s = find_intersection(q1, 0, q2, 0)
            ax.plot(s, parabola(s, q1), 'ro')
            ax.axvline(x=s, color='r', linestyle='--', alpha=0.5)

        ax.set_title(f'添加第{i}个抛物线后的包络线')
        ax.set_xlabel('x')
        ax.set_ylabel('距离值')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-1, 30)

    plt.tight_layout()
    plt.show()


# 绘制距离变换结果对比
def plot_distance_transform():
    # 创建一个简单的障碍物分布
    x = np.arange(20)
    obstacles = np.zeros_like(x)
    obstacles[[3, 10, 16]] = 1  # 在位置3,10,16设置障碍物

    # 计算距离变换（简化版）
    def simple_dt(obstacles):
        n = len(obstacles)
        dist = np.full(n, float('inf'))
        # 正向扫描
        last_obstacle = -float('inf')
        for i in range(n):
            if obstacles[i] == 1:
                last_obstacle = i
                dist[i] = 0
            else:
                dist[i] = min(dist[i], i - last_obstacle)
        # 反向扫描
        last_obstacle = float('inf')
        for i in range(n - 1, -1, -1):
            if obstacles[i] == 1:
                last_obstacle = i
                dist[i] = 0
            else:
                dist[i] = min(dist[i], last_obstacle - i)
        return dist ** 2  # 返回平方距离

    dist_sq = simple_dt(obstacles)

    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 绘制障碍物位置
    ax1.stem(x, obstacles, linefmt='r-', markerfmt='ro', basefmt='k-')
    ax1.set_title('障碍物位置')
    ax1.set_ylabel('是否为障碍物')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    # 绘制抛物线和距离变换结果
    for q in x[obstacles == 1]:
        ax2.plot(x, parabola(x, q), 'b-', alpha=0.5)
    ax2.plot(x, dist_sq, 'ro-', linewidth=2, label='距离变换结果（平方距离）')
    ax2.set_title('抛物线与距离变换结果对比')
    ax2.set_xlabel('位置 x')
    ax2.set_ylabel('平方距离')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-1, 40)

    plt.tight_layout()
    plt.show()


# 主函数，运行所有可视化
def main():
    # 绘制基本抛物线及其交点
    plot_basic_parabolas()

    # 绘制新抛物线加入的两种情况
    plot_addition_scenarios()

    # 绘制包络线形成过程
    plot_envelope_formation()

    # 绘制距离变换结果
    plot_distance_transform()


    print("1. 基本抛物线及其下包络线的形成")
    print("2. 新抛物线加入时的两种关键情况（s > z[k] 和 s <= z[k]）")
    print("3. 包络线随抛物线增加的逐步形成过程")
    print("4. 距离变换结果与抛物线的关系")


if __name__ == "__main__":
    main()
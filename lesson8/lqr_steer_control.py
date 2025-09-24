"""
路径跟踪仿真：使用LQR（线性二次调节器）进行转向控制，PID进行速度控制
"""
import scipy.linalg as la  # 用于线性代数运算（如矩阵求逆、特征值计算）
import matplotlib.pyplot as plt  # 用于绘图和动画显示
import math  # 用于数学运算（如三角函数、平方根）
import numpy as np  # 用于矩阵和数组操作
import sys  # 用于系统操作（如程序退出）
import pathlib  # 用于路径处理（本代码未直接使用，保留导入）
from pure_pursuit import plot_vehicle

from utils.angle import angle_mod  # 导入角度归一化工具函数
import cubic_spline_planner  # 导入三次样条路径规划器


# PID速度控制器参数
Kp = 1.0  # 速度比例增益（控制加速度与速度误差的比例）

# LQR控制器参数
Q = np.eye(4)  # 状态权重矩阵（调整各状态量在代价函数中的占比）
R = np.eye(1)  # 控制权重矩阵（调整控制量在代价函数中的占比）

# 车辆与仿真参数
dt = 0.1  # 时间步长 [s]（仿真迭代间隔）
L = 0.5  # 车辆轴距 [m]（前后轴距离）
max_steer = np.deg2rad(45.0)  # 最大转向角 [rad]（物理限制）

show_animation = True  # 是否显示实时仿真动画
# show_animation = False  # 关闭动画显示


class State:
    """
    车辆状态类：存储车辆当前的位置、姿态和运动状态
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x      # 车辆在全局坐标系中的x坐标 [m]
        self.y = y      # 车辆在全局坐标系中的y坐标 [m]
        self.yaw = yaw  # 车辆偏航角（航向角）[rad]（与x轴正方向的夹角）
        self.v = v      # 车辆前进速度 [m/s]


def update(state, a, delta):
    """
    根据车辆运动学模型更新车辆状态

    参数:
        state: State类实例，当前车辆状态（x, y, yaw, v）
        a: 加速度 [m/s²]（控制输入，正值加速，负值减速）
        delta: 转向角 [rad]（控制输入，正值左偏，负值右偏）

    返回:
        State类实例，更新后的车辆状态
    """
    # 限制转向角在物理允许范围内
    #if delta >= max_steer:
    #    delta = max_steer
    #if delta <= -max_steer:
    #    delta = -max_steer
    delta = np.clip(delta, -max_steer, max_steer)
    # 车辆运动学模型（基于自行车模型）
    # 位置更新：根据当前速度和航向角计算位移
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    # 偏航角更新：根据速度、轴距和转向角计算转向引起的角度变化
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    # 速度更新：根据加速度和时间步长更新
    state.v = state.v + a * dt

    return state


def pid_control(target, current):
    """
    PID速度控制器（此处仅使用比例控制）

    参数:
        target: 目标速度 [m/s]
        current: 当前速度 [m/s]

    返回:
        计算得到的加速度 [m/s²]（控制输入）
    """
    # 比例控制：加速度与速度误差成正比
    a = Kp * (target - current)

    return a


def pi_2_pi(angle):
    """
    将角度归一化到[-π, π]范围内（避免角度计算中的跳变）

    参数:
        angle: 输入角度 [rad]

    返回:
        归一化后的角度 [rad]
    """
    return angle_mod(angle)


def solve_DARE(A, B, Q, R):
    """
    求解离散时间代数黎卡提方程(DARE)，用于LQR控制器设计

    参数:
        A: 系统状态矩阵（离散时间）
        B: 系统输入矩阵（离散时间）
        Q: 状态权重矩阵（正定）
        R: 控制权重矩阵（正定）

    返回:
        X: 黎卡提方程的解（正定矩阵）
    """
    X = Q  # 初始化解矩阵
    Xn = Q  # 迭代更新的解矩阵
    max_iter = 150  # 最大迭代次数（防止不收敛）
    eps = 0.01      # 收敛精度（矩阵元素变化阈值）

    for i in range(max_iter):
        # 黎卡提方程迭代公式
        Xn = A.T @ X @ A - A.T @ X @ B @ la.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        # 判断是否收敛（最大元素变化小于精度阈值）
        if (abs(Xn - X)).max() < eps:
            break
        X = Xn  # 更新解矩阵

    return Xn


def dlqr(A, B, Q, R):
    """
    求解离散时间LQR控制器

    系统模型: x[k+1] = A x[k] + B u[k]
    代价函数: sum(x[k].T * Q * x[k] + u[k].T * R * u[k]) （最小化累积代价）

    参数:
        A: 系统状态矩阵（离散时间）
        B: 系统输入矩阵（离散时间）
        Q: 状态权重矩阵（惩罚偏离期望状态的代价）
        R: 控制权重矩阵（惩罚控制量大小的代价）

    返回:
        K: LQR反馈增益矩阵（u = -Kx）
        X: 黎卡提方程的解
        eigVals: 闭环系统特征值（用于稳定性分析）
    """

    # 求解黎卡提方程得到最优状态权重矩阵
    X = solve_DARE(A, B, Q, R)

    # 计算LQR反馈增益
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    # 计算闭环系统（x[k+1] = (A - B*K)x[k]）的特征值（判断稳定性）
    eigVals, eigVecs = la.eig(A - B @ K)

    return K, X, eigVals


def lqr_steering_control(state, cx, cy, cyaw, ck, pe, pth_e):
    """
    LQR转向控制器：计算车辆跟踪参考路径所需的转向角

    参数:
        state: State类实例，当前车辆状态
        cx: 参考路径的x坐标序列 [m]
        cy: 参考路径的y坐标序列 [m]
        cyaw: 参考路径的航向角序列 [rad]
        ck: 参考路径的曲率序列 [1/m]
        pe: 上一时刻的横向误差 [m]（车辆到路径的侧向距离）
        pth_e: 上一时刻的航向误差 [rad]（车辆航向与路径航向的偏差）

    返回:
        delta: 计算得到的转向角 [rad]
        ind: 参考路径上最近点的索引
        e: 当前横向误差 [m]
        th_e: 当前航向误差 [rad]
    """
    # 找到车辆当前位置在参考路径上的最近点及横向误差
    ind, e = calc_nearest_index(state, cx, cy, cyaw)

    k = ck[ind]     # 最近点处的路径曲率
    v = state.v     # 当前车辆速度
    # 计算当前航向误差（归一化到[-π, π]）
    th_e = pi_2_pi(state.yaw - cyaw[ind])

    # 构建系统状态矩阵A（离散时间）
    # 状态量：[横向误差, 横向误差导数, 航向误差, 航向误差导数]
    A = np.zeros((4, 4))
    A[0, 0] = 1.0
    A[0, 1] = dt  # 横向误差 = 上一时刻误差 + 误差导数*dt
    A[1, 2] = v   # 横向误差导数与航向误差相关（速度影响）
    A[2, 2] = 1.0
    A[2, 3] = dt  # 航向误差 = 上一时刻误差 + 误差导数*dt

    # 构建输入矩阵B（离散时间）：控制量（转向角）对状态的影响
    B = np.zeros((4, 1))
    B[3, 0] = v / L  # 航向误差导数与转向角相关（速度/轴距影响）

    # 求解LQR得到反馈增益矩阵K
    K, _, _ = dlqr(A, B, Q, R)

    # 构建当前状态向量x
    x = np.zeros((4, 1))
    x[0, 0] = e                     # 横向误差
    x[1, 0] = (e - pe) / dt         # 横向误差导数（数值近似）
    x[2, 0] = th_e                  # 航向误差
    x[3, 0] = (th_e - pth_e) / dt   # 航向误差导数（数值近似）

    # 前馈控制：基于路径曲率的期望转向角（抵消路径弯曲）
    ff = math.atan2(L * k, 1)  # 曲率对应的转向角（几何关系）
    # 反馈控制：基于LQR的误差修正（抵消跟踪偏差）
    fb = pi_2_pi((-K @ x)[0, 0])  # 负反馈（-Kx）

    # 总转向角 = 前馈控制 + 反馈控制
    delta = ff + fb

    return delta, ind, e, th_e


def calc_nearest_index(state, cx, cy, cyaw):
    """
    计算车辆当前位置到参考路径的最近点索引及横向误差（带方向）

    参数:
        state: State类实例，当前车辆状态
        cx: 参考路径的x坐标序列 [m]
        cy: 参考路径的y坐标序列 [m]
        cyaw: 参考路径的航向角序列 [rad]

    返回:
        ind: 最近点在路径序列中的索引
        mind: 横向误差 [m]（路径右侧为正，左侧为负）
    """
    # 计算车辆到每个路径点的x、y方向距离
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    # 计算距离的平方（避免开方运算，提高效率）
    d = [idx **2 + idy** 2 for (idx, idy) in zip(dx, dy)]

    # 找到最近点的索引和距离
    mind = min(d)  # 最小距离的平方
    ind = d.index(mind)  # 最近点索引
    mind = math.sqrt(mind)  # 实际距离（开方）, 排序后计算

    # 计算最近点到车辆的相对位置（路径点 - 车辆位置）
    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    # 计算路径航向与车辆到路径点连线的夹角（判断误差方向）
    # 若夹角 < 0，说明车辆在路径左侧，误差取负值
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1  # 左侧误差为负，右侧为正

    return ind, mind


def closed_loop_prediction(cx, cy, cyaw, ck, speed_profile, goal):
    """
    闭环仿真主函数：实现车辆路径跟踪的全过程仿真

    参数:
        cx: 参考路径的x坐标序列 [m]
        cy: 参考路径的y坐标序列 [m]
        cyaw: 参考路径的航向角序列 [rad]
        ck: 参考路径的曲率序列 [1/m]
        speed_profile: 沿路径的目标速度序列 [m/s]
        goal: 目标点坐标 (x, y) [m]

    返回:
        t: 时间序列 [s]
        x: 车辆x坐标序列 [m]
        y: 车辆y坐标序列 [m]
        yaw: 车辆偏航角序列 [rad]
        v: 车辆速度序列 [m/s]
    """
    T = 500.0  # 最大仿真时间 [s]（防止无限循环）
    goal_dis = 0.3  # 到达目标的距离阈值 [m]（小于此值视为到达）
    stop_speed = 0.05  # 停止速度阈值 [m/s]（小于此值视为停止）

    # 初始化车辆状态（起点位置、航向、速度）
    state = State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)

    # 存储仿真数据的列表
    time = 0.0  # 当前时间
    x = [state.x]  # x坐标记录
    y = [state.y]  # y坐标记录
    yaw = [state.yaw]  # 偏航角记录
    v = [state.v]  # 速度记录
    t = [0.0]  # 时间记录

    # 初始化误差（上一时刻的横向误差和航向误差）
    e, e_th = 0.0, 0.0

    # 主仿真循环（未超时且未到达目标）
    while T >= time:
        # LQR转向控制：计算转向角，更新最近点索引和误差
        dl, target_ind, e, e_th = lqr_steering_control(state, cx, cy, cyaw, ck, e, e_th)

        # PID速度控制：根据目标速度和当前速度计算加速度
        ai = pid_control(speed_profile[target_ind], state.v)
        # 更新车辆状态
        state = update(state, ai, dl)

        # 若车辆接近停止，提前更新目标点索引（避免卡在原地）
        if abs(state.v) <= stop_speed:
            target_ind += 1

        # 累加时间
        time = time + dt

        # 检查是否到达目标点
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.hypot(dx, dy) <= goal_dis:
            print("到达目标点")
            break

        # 记录当前状态
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        # 实时显示动画（每步更新）
        if target_ind % 1 == 0 and show_animation:
            plt.cla()  # 清除当前图像
            # 绑定ESC键退出仿真
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "-r", label="waypoints")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="Target")
            plt.axis("equal")  # 等比例显示
            plot_vehicle(state.x, state.y, state.yaw, dl, is_reverse=False)
            plt.grid(True)  # 显示网格
            plt.title(f"Speed[km/h]: {round(state.v * 3.6, 2)}, Target idx: {target_ind}")
            plt.pause(0.0001)  # 短暂暂停以显示图像

    return t, x, y, yaw, v


def calc_speed_profile(cx, cy, cyaw, target_speed):
    """
    计算速度剖面：根据参考路径的航向变化调整目标速度（转向处减速）

    参数:
        cx: 参考路径的x坐标序列 [m]
        cy: 参考路径的y坐标序列 [m]
        cyaw: 参考路径的航向角序列 [rad]
        target_speed: 基础目标速度 [m/s]（直线路段的目标速度）

    返回:
        speed_profile: 沿路径的目标速度序列 [m/s]
    """
    # 初始化速度剖面（默认全路段为基础目标速度）
    speed_profile = [target_speed] * len(cx)

    direction = 1.0  # 方向标志（1.0：前进，-1.0：后退，用于判断转向）

    # 遍历路径点，根据航向角变化设置转向点速度
    for i in range(len(cx) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])  # 相邻点的航向角变化量
        # 判断是否为显著转向点（航向角变化在45°到90°之间）
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1  # 转向时反转方向标志

        # 根据方向标志设置目标速度（后退时为负速度）
        if direction != 1.0:
            speed_profile[i] = -target_speed
        else:
            speed_profile[i] = target_speed

        # 转向点处速度设为0（短暂停止以完成转向）
        if switch:
            speed_profile[i] = 0.0

    # 终点速度设为0（到达目标后停止）
    #speed_profile[-1] = 0.0
    # 接近终点时逐渐减速
    for i in range(40):
        speed_profile[-i] = target_speed / (50 - i)
        # 最低速度限制（1 km/h转换为m/s）
        if speed_profile[-i] <= 1.0 / 3.6:
            speed_profile[-i] = 1.0 / 3.6

    return speed_profile


def main():
    """程序主函数：设置参数、生成路径、执行仿真并显示结果"""
    print("LQR转向控制路径跟踪开始!!")
    # 参考路径的控制点（通过样条插值生成平滑路径）
    ax = [0.0, 6.0, 12.5, 10.0, 7.5, 3.0, -1.0]  # 控制点x坐标
    ay = [0.0, -3.0, -5.0, 6.5, 3.0, 5.0, -2.0]  # 控制点y坐标
    goal = [ax[-1], ay[-1]]  # 目标点（最后一个控制点）

    # 用三次样条插值生成平滑参考路径
    # ds：路径点间隔 [m]，返回路径的x、y坐标、航向角、曲率、路径长度
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    target_speed = 10.0 / 3.6  # 基础目标速度（10km/h转换为m/s）

    # 计算速度剖面（根据路径曲率调整目标速度）
    sp = calc_speed_profile(cx, cy, cyaw, target_speed)

    # 执行闭环仿真
    t, x, y, yaw, v = closed_loop_prediction(cx, cy, cyaw, ck, sp, goal)

    # 显示最终结果图表（若开启动画）
    if show_animation:
        plt.close()  # 关闭实时仿真窗口

        # 路径跟踪结果对比图
        plt.subplots(1)
        plt.plot(ax, ay, "xb", label="control points")
        plt.plot(cx, cy, "-r", label="Path")
        plt.plot(x, y, "-g", label="Trajectory")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        # 路径航向角曲线图
        plt.subplots(1)
        plt.plot(s, [np.rad2deg(iyaw) for iyaw in cyaw], "-r", label="Heading angle")
        plt.grid(True)
        plt.legend()
        plt.xlabel("Path[m]")
        plt.ylabel("Heading[deg]")

        # 路径曲率曲线图
        plt.subplots(1)
        plt.plot(s, ck, "-r", label="curvature")
        plt.grid(True)
        plt.legend()
        plt.xlabel("path[m]")
        plt.ylabel("curvature [1/m]")

        plt.show()  # 显示所有图表


if __name__ == '__main__':
    main()  # 程序入口
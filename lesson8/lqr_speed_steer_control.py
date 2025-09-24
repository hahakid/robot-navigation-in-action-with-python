"""
Path tracking simulation with LQR speed and steering control
"""
import math
import sys
import matplotlib.pyplot as plt  # 用于绘图显示轨迹和速度等信息
import numpy as np  # 用于数值计算和矩阵运算
import scipy.linalg as la  # 用于线性代数运算（如矩阵求逆、特征值计算）
import pathlib

from utils.angle import angle_mod  # 用于角度归一化处理
import cubic_spline_planner  # 用于生成三次样条路径

# === 参数设置 =====

# LQR参数：状态权重矩阵和输入权重矩阵
lqr_Q = np.eye(5)  # 状态向量的权重矩阵，单位矩阵表示各状态权重相同
lqr_R = np.eye(2)  # 控制输入的权重矩阵，单位矩阵表示各输入权重相同
dt = 0.1  # 时间步长 [s]
L = 0.5  # 车辆轴距 [m]
max_steer = np.deg2rad(45.0)  # 最大转向角 [rad]，限制转向角度范围

show_animation = True  # 是否显示动画


class State:
    """车辆状态类，存储车辆的位置、姿态和速度信息"""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x  # x坐标 [m]
        self.y = y  # y坐标 [m]
        self.yaw = yaw  # 偏航角（航向角）[rad]
        self.v = v  # 速度 [m/s]


def update(state, a, delta):
    """
    根据控制输入更新车辆状态
    :param state: 当前车辆状态
    :param a: 加速度 [m/s²]
    :param delta: 转向角 [rad]
    :return: 更新后的车辆状态
    """
    # 限制转向角在最大范围内
    #if delta >= max_steer:
    #    delta = max_steer
    #if delta <= -max_steer:
    #    delta = -max_steer
    # 使用np.clip函数将转向角限制在[-max_steer, max_steer]范围内
    delta = np.clip(delta, -max_steer, max_steer)
    # 基于运动学模型更新状态
    # 位置x更新：速度在x方向的投影乘以时间步长
    # 计算x方向位移 = 当前x坐标 + 速度 * cos(偏航角) * 时间步长
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    # 位置y更新：速度在y方向的投影乘以时间步长
    # 计算y方向位移 = 当前y坐标 + 速度 * sin(偏航角) * 时间步长
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    # 偏航角更新：基于阿克曼转向模型，速度/轴距 * tan(转向角) 为角速度，乘以时间步长
    # 计算偏航角变化 = 当前偏航角 + (速度/轴距) * tan(转向角) * 时间步长
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    # 速度更新：加速度乘以时间步长
    # 计算新速度 = 当前速度 + 加速度 * 时间步长
    state.v = state.v + a * dt

    return state


def pi_2_pi(angle):
    """将角度归一化到[-π, π]范围内"""
    return angle_mod(angle)


def solve_dare(A, B, Q, R):
    """
    求解离散时间代数黎卡提方程(DARE)
    :param A: 系统状态矩阵
    :param B: 系统输入矩阵
    :param Q: 状态权重矩阵
    :param R: 输入权重矩阵
    :return: 黎卡提方程的解X
    """
    x = Q  # 初始化X为Q
    x_next = Q
    max_iter = 150  # 最大迭代次数
    eps = 0.01  # 收敛阈值

    # 迭代求解DARE
    for i in range(max_iter):
        # 黎卡提方程迭代公式
        x_next = A.T @ x @ A - A.T @ x @ B @ \
                 la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
        # 检查是否收敛
        if (abs(x_next - x)).max() < eps:
            break
        x = x_next

    return x_next


def dlqr(A, B, Q, R):
    """
    求解离散时间LQR控制器
    系统模型：x[k+1] = A x[k] + B u[k]
    代价函数：sum(x[k].T*Q*x[k] + u[k].T*R*u[k])
    :param A: 状态矩阵
    :param B: 输入矩阵
    :param Q: 状态权重矩阵
    :param R: 输入权重矩阵
    :return: LQR增益矩阵K，黎卡提方程解X，闭环系统特征值
    """
    # 求解黎卡提方程
    X = solve_dare(A, B, Q, R)

    # 计算LQR增益K
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    # 计算闭环系统的特征值（用于稳定性分析）
    eig_result = la.eig(A - B @ K)

    return K, X, eig_result[0]


def lqr_speed_steering_control(state, cx, cy, cyaw, ck, pe, pth_e, sp, Q, R):
    """
    LQR速度和转向控制函数，计算转向角和加速度
    :param state: 当前车辆状态
    :param cx: 路径x坐标列表
    :param cy: 路径y坐标列表
    :param cyaw: 路径各点的偏航角列表
    :param ck: 路径各点的曲率列表
    :param pe: 上一时刻的横向误差
    :param pth_e: 上一时刻的角度误差
    :param sp: 速度剖面（目标速度列表）
    :param Q: LQR状态权重矩阵
    :param R: LQR输入权重矩阵
    :return: 转向角、目标点索引、当前横向误差、当前角度误差、加速度
    """
    # 计算当前车辆在路径上的最近点索引和横向误差
    ind, e = calc_nearest_index(state, cx, cy, cyaw)

    # 获取目标速度（当前最近点对应的速度）
    tv = sp[ind]

    # 获取当前路径点的曲率和车辆当前速度
    k = ck[ind]
    v = state.v
    # 计算角度误差（车辆当前偏航角与路径偏航角的差值，归一化到[-π, π]）
    th_e = pi_2_pi(state.yaw - cyaw[ind])

    # 构造系统状态矩阵A（5x5）
    # 状态向量：[横向误差e, e的导数, 角度误差th_e, th_e的导数, 速度差delta_v]
    A = np.zeros((5, 5))
    A[0, 0] = 1.0
    A[0, 1] = dt  # e的更新与e的导数相关
    A[1, 2] = v   # e的导数与角度误差相关
    A[2, 2] = 1.0
    A[2, 3] = dt  # th_e的更新与th_e的导数相关
    A[4, 4] = 1.0  # 速度差的状态转移

    # 构造系统输入矩阵B（5x2）
    # 控制输入：[转向角delta, 加速度a]
    B = np.zeros((5, 2))
    B[3, 0] = v / L  # th_e的导数与转向角相关（基于运动学模型）
    B[4, 1] = dt     # 速度差的变化与加速度相关

    # 求解LQR，得到增益矩阵K
    K, _, _ = dlqr(A, B, Q, R)

    # 构造状态向量x
    x = np.zeros((5, 1))
    x[0, 0] = e  # 横向误差
    x[1, 0] = (e - pe) / dt  # 横向误差的导数（用前向差分近似）
    x[2, 0] = th_e  # 角度误差
    x[3, 0] = (th_e - pth_e) / dt  # 角度误差的导数（前向差分近似）
    x[4, 0] = v - tv  # 速度差（当前速度-目标速度）

    # 计算最优控制输入（u* = -Kx）
    ustar = -K @ x

    # 计算转向角：前馈控制（基于路径曲率）+ 反馈控制（LQR输出）
    ff = math.atan2(L * k, 1)  # 前馈转向角（根据曲率计算，补偿路径弯曲）
    fb = pi_2_pi(ustar[0, 0])  # 反馈转向角（LQR计算的修正量）
    delta = ff + fb  # 总转向角

    # 计算加速度（LQR输出的速度修正量）
    accel = ustar[1, 0]

    return delta, ind, e, th_e, accel


def calc_nearest_index(state, cx, cy, cyaw):
    """
    计算车辆当前位置到路径上最近点的索引和横向误差
    :param state: 当前车辆状态
    :param cx: 路径x坐标列表
    :param cy: 路径y坐标列表
    :param cyaw: 路径各点的偏航角列表
    :return: 最近点索引、横向误差（带符号，左侧为正，右侧为负）
    """
    # 计算车辆到路径各点的x、y方向距离
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    # 计算距离的平方（避免开方，提高计算效率）
    d = [idx **2 + idy** 2 for (idx, idy) in zip(dx, dy)]

    # 找到最近点的距离和索引
    mind = min(d)
    ind = d.index(mind)
    mind = math.sqrt(mind)  # 实际距离

    # 计算最近点到车辆的向量（路径点 - 车辆位置）
    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    # 通过角度判断车辆在路径的左侧还是右侧，调整误差符号
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1  # 右侧时误差为负

    return ind, mind


def do_simulation(cx, cy, cyaw, ck, speed_profile, goal):
    """
    进行路径跟踪仿真
    :param cx: 路径x坐标列表
    :param cy: 路径y坐标列表
    :param cyaw: 路径偏航角列表
    :param ck: 路径曲率列表
    :param speed_profile: 速度剖面（目标速度列表）
    :param goal: 目标点坐标
    :return: 时间列表、x轨迹、y轨迹、偏航角轨迹、速度轨迹
    """
    T = 500.0  # 最大仿真时间 [s]
    goal_dis = 0.3  # 到达目标的距离阈值 [m]
    stop_speed = 0.05  # 停止的速度阈值 [m/s]

    # 初始化车辆状态（起点）
    state = State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)

    # 记录仿真数据的列表
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]

    # 初始化误差
    e, e_th = 0.0, 0.0

    # 仿真主循环
    while T >= time:
        # 调用LQR控制函数，获取转向角、加速度和误差信息
        # return 转向角、目标点索引、当前横向误差、当前角度误差、加速度
        dl, target_ind, e, e_th, ai = lqr_speed_steering_control(
            state, cx, cy, cyaw, ck, e, e_th, speed_profile, lqr_Q, lqr_R)

        # 更新车辆状态
        state = update(state, ai, dl)

        # 当速度很低时，提前更新目标点索引（避免在终点附近震荡）
        if abs(state.v) <= stop_speed:
            target_ind += 1

        # 更新时间
        time = time + dt

        # 检查是否到达目标
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.hypot(dx, dy) <= goal_dis:
            print("Goal")
            break

        # 记录轨迹数据
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        # 实时显示动画
        if target_ind % 1 == 0 and show_animation:
            plt.cla()
            # 按ESC键退出仿真
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "-r", label="course")  # 目标路径
            plt.plot(x, y, "ob", label="trajectory")  # 实际轨迹
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")  # 当前目标点
            plt.axis("equal")
            plt.grid(True)
            plt.title(f"speed[km/h]: {round(state.v * 3.6, 2)}, target index: {target_ind}")
            plt.pause(0.0001)

    return t, x, y, yaw, v


def calc_speed_profile(cyaw, target_speed):
    """
    计算速度剖面（根据路径曲率调整目标速度）
    :param cyaw: 路径各点的偏航角列表
    :param target_speed: 基准目标速度 [m/s]
    :return: 调整后的速度剖面列表
    """
    speed_profile = [target_speed] * len(cyaw)  # 初始化速度剖面为基准速度

    direction = 1.0  # 方向标志（用于检测转向）

    # 根据路径偏航角变化设置转向时的速度（减速）
    for i in range(len(cyaw) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])  # 相邻点的偏航角差
        # 判断是否为转向点（偏航角变化在π/4到π/2之间）
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1  # 转向时切换方向标志

        # 转向时减速（此处设置为反向速度，实际通过绝对值体现减速）
        if direction != 1.0:
            speed_profile[i] = -target_speed
        else:
            speed_profile[i] = target_speed

        # 转向点处速度设置为0（短暂停车）
        if switch:
            speed_profile[i] = 0.0

    # 接近终点时逐渐减速
    for i in range(40):
        speed_profile[-i] = target_speed / (50 - i)
        # 最低速度限制（1 km/h转换为m/s）
        if speed_profile[-i] <= 1.0 / 3.6:
            speed_profile[-i] = 1.0 / 3.6

    return speed_profile


def main():
    """主函数：生成路径、设置参数、运行仿真并显示结果"""
    print("LQR steering control tracking start!!")
    # 路径waypoints（控制点）
    ax = [0.0, 6.0, 12.5, 10.0, 17.5, 20.0, 25.0]
    ay = [0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0]
    goal = [ax[-1], ay[-1]]  # 目标点为最后一个waypoint

    # 用三次样条插值生成平滑路径
    # ds为路径点间隔 [m]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    target_speed = 10.0 / 3.6  # 目标速度（10 km/h转换为m/s）

    # 计算速度剖面
    sp = calc_speed_profile(cyaw, target_speed)

    # 运行仿真
    t, x, y, yaw, v = do_simulation(cx, cy, cyaw, ck, sp, goal)

    # 显示最终结果图表
    if show_animation:  # pragma: no cover
        plt.close()
        # 轨迹对比图
        plt.subplots(1)
        plt.plot(ax, ay, "xb", label="waypoints")  # 控制点
        plt.plot(cx, cy, "-r", label="target course")  # 目标路径
        plt.plot(x, y, "-g", label="tracking")  # 跟踪轨迹
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        # 速度曲线图
        plt.subplots(1)
        plt.plot(t, np.array(v)*3.6, label="speed")  # 转换为km/h
        plt.grid(True)
        plt.xlabel("Time [sec]")
        plt.ylabel("Speed [km/h]")
        plt.legend()

        # 路径偏航角曲线图
        plt.subplots(1)
        plt.plot(s, [np.rad2deg(iyaw) for iyaw in cyaw], "-r", label="yaw")  # 转换为度
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("yaw angle[deg]")

        # 路径曲率曲线图
        plt.subplots(1)
        plt.plot(s, ck, "-r", label="curvature")
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("curvature [1/m]")

        plt.show()


if __name__ == '__main__':
    main()
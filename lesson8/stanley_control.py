"""
使用Stanley转向控制和PID速度控制的路径跟踪仿真。
参考资料：
    - [Stanley：赢得DARPA挑战赛的机器人](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [自主车辆路径跟踪](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)
"""
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图可视化

from utils.angle import angle_mod  # 导入角度处理工具函数
import cubic_spline_planner  # 导入三次样条路径规划器
from utils.plot import get_frame_as_array
import imageio
from pure_pursuit import plot_vehicle

save_gif = True  # True  # 是否保存GIF
frames = []  # 存储GIF帧的列表
name = "stanley.gif"


k = 0.5  # Stanley控制增益（用于横向误差校正）
Kp = 1.0  # 速度PID控制器的比例增益
dt = 0.1  # 时间步长 [秒]
L = 2.9  # 车辆轴距 [米]
max_steer = np.radians(30.0)  # 最大转向角 [弧度]（限制为30度）

show_animation = True  # 是否显示仿真动画


class State:
    """
    表示车辆状态的类

    :param x: (float) x坐标
    :param y: (float) y坐标
    :param yaw: (float) 偏航角（航向角，弧度）
    :param v: (float) 速度（米/秒）
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """初始化车辆状态对象"""
        super().__init__()
        self.x = x  # x坐标
        self.y = y  # y坐标
        self.yaw = yaw  # 偏航角（航向角）
        self.v = v  # 行驶速度

    def update(self, acceleration, delta):
        """
        更新车辆状态（基于自行车模型）

        :param acceleration: (float) 加速度（控制输入，影响速度）
        :param delta: (float) 转向角（控制输入，影响航向）
        """
        # 限制转向角在最大范围内（防止过度转向）
        delta = np.clip(delta, -max_steer, max_steer)

        # 基于自行车模型更新位置和姿态
        self.x += self.v * np.cos(self.yaw) * dt  # x方向位移 = 速度×cos(航向)×时间
        self.y += self.v * np.sin(self.yaw) * dt  # y方向位移 = 速度×sin(航向)×时间
        self.yaw += self.v / L * np.tan(delta) * dt  # 航向变化 = (速度/轴距)×tan(转向角)×时间
        self.yaw = normalize_angle(self.yaw)  # 将航向角归一化到[-π, π]
        self.v += acceleration * dt  # 速度变化 = 加速度×时间


def pid_control(target, current):
    """
    速度比例控制（PID中的P环）

    :param target: (float) 目标速度（米/秒）
    :param current: (float) 当前速度（米/秒）
    :return: (float) 计算得到的加速度（控制量）
    """
    return Kp * (target - current)  # 加速度 = 比例增益×(目标速度-当前速度)


def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley转向控制算法（核心函数）

    :param state: (State对象) 车辆当前状态
    :param cx: ([float]) 参考路径的x坐标列表
    :param cy: ([float]) 参考路径的y坐标列表
    :param cyaw: ([float]) 参考路径各点的航向角列表
    :param last_target_idx: (int) 上一时刻跟踪的路径点索引
    :return: (float, int) 计算得到的转向角和当前跟踪的路径点索引
    """
    # 计算当前应跟踪的路径点索引和前轴横向误差
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    # 防止目标点索引回退（确保车辆沿路径前进）
    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # 航向误差：参考路径目标点航向与车辆当前航向的差值（归一化处理）
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # 横向误差校正角：基于横向误差和当前速度计算（k为比例系数）
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # 总转向角 = 航向误差 + 横向误差校正角
    delta = theta_e + theta_d

    return delta, current_target_idx


def normalize_angle(angle):
    """
    将角度归一化到[-π, π]范围

    :param angle: (float) 输入角度（弧度）
    :return: (float) 归一化后的角度（弧度）
    """
    return angle_mod(angle)  # 调用工具函数实现角度归一化


def calc_target_index(state, cx, cy):
    """
    计算参考路径中距离车辆前轴最近的目标点索引

    :param state: (State对象) 车辆当前状态
    :param cx: [float] 参考路径x坐标列表
    :param cy: [float] 参考路径y坐标列表
    :return: (int, float) 目标点索引和前轴横向误差
    """
    # 计算车辆前轴位置（基于当前位置和航向）
    fx = state.x + L * np.cos(state.yaw)  # 前轴x坐标 = 车辆x + 轴距×cos(航向)
    fy = state.y + L * np.sin(state.yaw)  # 前轴y坐标 = 车辆y + 轴距×sin(航向)

    # 计算前轴到路径上所有点的距离
    dx = [fx - icx for icx in cx]  # x方向距离差列表
    dy = [fy - icy for icy in cy]  # y方向距离差列表
    d = np.hypot(dx, dy)  # 欧氏距离列表（前轴到每个路径点的直线距离）
    target_idx = np.argmin(d)  # 距离最小的点索引（初始目标点）

    # 计算前轴横向误差：将距离投影到前轴垂直方向（左侧为正）
    # 前轴垂直向量（指向车辆左侧）
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2), -np.sin(state.yaw + np.pi / 2)]
    # 横向误差 = 目标点与前轴的向量 与 前轴垂直向量 的点积
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle


def main():
    """主函数：基于三次样条路径演示Stanley转向控制效果"""
    # 目标路径控制点（定义路径的关键节点）
    axx = [0.0, 100.0, 100.0, 50.0, 60.0]  # x方向控制点
    ayy = [0.0, 0.0, -30.0, -20.0, 0.0]   # y方向控制点

    # 使用三次样条生成平滑路径
    # cx, cy: 路径点坐标；cyaw: 路径点航向角；ck: 曲率；s: 路径长度
    cx, cy, cyaw, ck, _ = cubic_spline_planner.calc_spline_course(axx, ayy, ds=0.1)

    target_speed = 30.0 / 3.6  # 目标速度 [米/秒]（30千米/小时转换为米/秒）

    max_simulation_time = 100.0  # 最大仿真时间 [秒]

    # 初始状态（车辆起始位置、航向和速度）
    state = State(x=-0.0, y=5.0, yaw=np.radians(20.0), v=0.0)

    last_idx = len(cx) - 1  # 路径最后一个点的索引（终点）
    time = 0.0  # 仿真时间计数器

    # 记录仿真过程中的状态数据（用于后续绘图）
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]

    # 初始化目标点索引
    target_idx, _ = calc_target_index(state, cx, cy)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_xlim(min(cx) - 10, max(cx) + 10)
    ax.set_ylim(min(cy) - 10, max(cy) + 10)
    plt.ion()  # 开启交互模式

    # 仿真主循环：直到到达终点或超过最大仿真时间
    while max_simulation_time >= time and last_idx > target_idx:
        # 计算速度控制量
        ai = pid_control(target_speed, state.v)
        # 计算转向控制量（转向角）并更新目标点索引
        di, target_idx = stanley_control(state, cx, cy, cyaw, target_idx)
        # 更新车辆状态
        state.update(ai, di)

        # 累加时间
        time += dt

        # 记录状态数据
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        # 实时动画显示
        if show_animation:
            ax.cla()  # 清除当前图像
            # 按ESC键退出仿真
            #plt.gcf().canvas.mpl_connect('key_release_event',
            #        lambda event: [exit(0) if event.key == 'escape' else None])
            ax.plot(cx, cy, ".r", label="course")  # 绘制参考路径
            ax.plot(axx, ayy, "ok", label="waypoints", markersize=10)
            ax.plot(x, y, "-b", label="trajectory")    # 绘制车辆轨迹
            ax.plot(cx[target_idx], cy[target_idx], "xg", label="Target")  # 绘制当前目标点
            plot_vehicle(state.x, state.y, state.yaw, di, is_reverse=False)
            ax.axis("equal")  # 等比例坐标
            ax.grid(True)     # 显示网格
            plt.title(f"speed: [km/h]: {state.v * 3.6:.1f}")  # 显示当前速度（转换为km/h）

            if show_animation:
                plt.pause(0.001)   # 暂停一小段时间，更新图像

            if save_gif:
                fig.canvas.draw()
                frame = get_frame_as_array(fig)
                frames.append(frame)

    # 仿真结束后验证：确保到达终点
    assert last_idx >= target_idx, "Cannot reach goal"

    # 仿真结束后绘制最终结果
    if show_animation:
        # 绘制路径和轨迹对比
        ax.plot(cx, cy, ".r", label="course")
        ax.plot(x, y, "-b", label="trajectory")
        ax.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        ax.axis("equal")
        ax.grid(True)

        if save_gif:
            fig.canvas.draw()
            frame = get_frame_as_array(fig)
            frames.append(frame)

    if save_gif and len(frames) > 0:
        imageio.mimsave(name, frames, fps=15, loop=1)
        print(f"GIF saved to {name}")

    plt.ioff()

    if show_animation:
        # 绘制速度随时间变化曲线
        plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")  # 转换为km/h
        plt.xlabel("Times[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()  # 显示图像

if __name__ == '__main__':
    main()  # 运行主函数
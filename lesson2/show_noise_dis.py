import numpy as np
import matplotlib.pyplot as plt

# 真实位置
def GPS_noise():
    true_pos = np.array([[0], [0]])
    GPS_NOISE = np.diag([0.5, 0.5]) ** 2

    # 生成1000次带噪声的GPS测量
    noise = np.sqrt(GPS_NOISE) @ np.random.randn(2, 1000)
    measurements = true_pos + noise

    # 绘制分布
    plt.figure(figsize=(6, 6))
    plt.scatter(measurements[0, :], measurements[1, :], s=5, alpha=0.5)
    plt.title('GPS measurements with noise')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u

def agent_control_noise():
    # 控制输入
    u = calc_input()

    # 噪声协方差矩阵
    INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
    STD_NOISE = np.sqrt(INPUT_NOISE)  # 标准差矩阵

    # 生成1000次带噪声的控制输入
    noise = STD_NOISE @ np.random.randn(2, 1000)
    ud_samples = u + noise

    # 绘制分布
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(ud_samples[0, :], bins=30, alpha=0.7)
    plt.axvline(u[0, 0], color='r', linestyle='--', label='ideal')
    plt.title('vn')
    plt.xlabel('v (m/s)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(np.rad2deg(ud_samples[1, :]), bins=30, alpha=0.7)
    plt.axvline(np.rad2deg(u[1, 0]), color='r', linestyle='--', label='ideal')
    plt.title('avn')
    plt.xlabel('angle v (°/s)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

agent_control_noise()
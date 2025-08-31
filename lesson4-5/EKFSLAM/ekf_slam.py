import math

import matplotlib.pyplot as plt
import numpy as np
from utils.angle import angle_mod
from utils.angle import rot_mat_2d
from utils.plot import get_frame_as_array
import imageio
save_gif = True
save_path = './ekf_slam.gif'

# EKF state covariance
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2

#  Simulation parameter
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2

DT = 0.5 # 0.1  # time tick [s]
SIM_TIME = 63.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range


M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

show_animation = True


def ekf_slam(xEst, PEst, u, z):
    # Predict
    G, Fx = jacob_motion(xEst, u)
    xEst[0:STATE_SIZE] = motion_model(xEst[0:STATE_SIZE], u)
    PEst = G.T @ PEst @ G + Fx.T @ Cx @ Fx
    initP = np.eye(2)

    # Update
    for iz in range(len(z[:, 0])):  # for each observation
        min_id = search_correspond_landmark_id(xEst, PEst, z[iz, 0:2])

        nLM = calc_n_lm(xEst)
        if min_id == nLM:  # 新路标
            print("New LM")
            # Extend state and covariance matrix
            xAug = np.vstack((xEst, calc_landmark_position(xEst, z[iz, :])))
            PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                              np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
            xEst = xAug
            PEst = PAug
        lm = get_landmark_position_from_state(xEst, min_id)
        y, S, H = calc_innovation(lm, xEst, PEst, z[iz, 0:2], min_id)

        K = (PEst @ H.T) @ np.linalg.inv(S)
        xEst = xEst + (K @ y)
        PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

    xEst[2] = pi_2_pi(xEst[2])  # 通过取模，归一化到 [-pi, pi]

    return xEst, PEst

# 匀速圆周运动
def calc_input():
    v = 1.0  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u

# 观测模型
def observation(xTrue, xd, u, RFID):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = np.zeros((0, 3))

    for i in range(len(RFID[:, 0])):

        dx = RFID[i, 0] - xTrue[0, 0]
        dy = RFID[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise
            angle_n = angle + np.random.randn() * Q_sim[1, 1] ** 0.5  # add noise
            zi = np.array([dn, angle_n, i])
            z = np.vstack((z, zi))

    # add noise to input
    ud = np.array([[
        u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5,
        u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T

    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = (F @ x) + (B @ u)
    return x


def calc_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def jacob_motion(x, u):
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros((STATE_SIZE, LM_SIZE * calc_n_lm(x)))))

    jF = np.array([[0.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0])],
                   [0.0, 0.0, DT * u[0, 0] * math.cos(x[2, 0])],
                   [0.0, 0.0, 0.0]], dtype=float)

    G = np.eye(len(x)) + Fx.T @ jF @ Fx

    return G, Fx,


def calc_landmark_position(x, z):
    zp = np.zeros((2, 1))

    zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
    zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

    return zp


def get_landmark_position_from_state(x, ind):
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]

    return lm


def search_correspond_landmark_id(xAug, PAug, zi):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = calc_n_lm(xAug)

    min_dist = []

    for i in range(nLM):
        lm = get_landmark_position_from_state(xAug, i)
        y, S, H = calc_innovation(lm, xAug, PAug, zi, i)
        min_dist.append(y.T @ np.linalg.inv(S) @ y)  # 马氏距离平方，考虑协方差方向

    min_dist.append(M_DIST_TH)  # new landmark

    min_id = min_dist.index(min(min_dist))

    return min_id


def calc_innovation(lm, xEst, PEst, z, LMid):
    delta = lm - xEst[0:2]  # 到路标的距离[d_x. d_y]
    q = (delta.T @ delta)[0, 0]  # 距离 d=sqrt(d_x^2 + d_y^2)
    z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]  # 路标角度
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])  # 路标观测值=[距离，角度]
    y = (z - zp).T  # 观测值与估计值之差
    y[1] = pi_2_pi(y[1])  #  角度归一化
    H = jacob_h(q, delta, xEst, LMid + 1)  #
    S = H @ PEst @ H.T + Cx[0:2, 0:2]

    return y, S, H

# 观测状态的 雅各比矩阵
def jacob_h(q, delta, x, i):
    """
    计算雅可比矩阵H的函数
    参数:
    q : 标量值，用于计算矩阵元素
    delta : 2x1的矩阵，用于计算矩阵元素
    x : 输入向量，用于计算nLM
    i : 整数，用于构建F2矩阵的维度
    返回:
    H : 计算得到的雅可比矩阵
    """
    # 计算q的平方根
    sq = math.sqrt(q)
    # 构建G矩阵，使用delta和q的值
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

    # 将G矩阵除以q进行归一化
    G = G / q
    # 计算nLM的值
    nLM = calc_n_lm(x)
    # 构建F1矩阵，由3x3的单位矩阵和3x(2*nLM)的零矩阵水平堆叠而成
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    # 构建F2矩阵，由不同维度的零矩阵、2x2的单位矩阵和零矩阵水平堆叠而成
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

    # 将F1和F2矩阵垂直堆叠成F矩阵
    F = np.vstack((F1, F2))

    # 计算最终的雅可比矩阵H，即G与F的矩阵乘积
    H = G @ F

    return H


def pi_2_pi(angle):
    return angle_mod(angle)


def main():
    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[10.0, -2.0],
                     [15.0, 10.0],
                     [3.0, 15.0],
                     [-5.0, 20.0]])

    # State Vector [x y yaw v]'
    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)

    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    # save gif
    frames = []
    fig = None
    ax = None
    if save_gif:
        fig = plt.figure(figsize=(6, 6), dpi=80)
        ax = fig.add_subplot(111)


    while SIM_TIME >= time:
        time += DT
        u = calc_input()
        #
        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID)
        # 经过观测模型，ud=带噪运动控制指令，z为带噪观测结果
        # xEst= 当前估计状态= [运动状态，观测到的RF_ID]
        # pEst= 状态协方差矩阵，对应 xEst维度^2
        xEst, PEst = ekf_slam(xEst, PEst, ud, z)

        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # pragma: no cover
            #plt.cla()
            if save_gif:  #清空画布
                ax.cla()
            else:
                plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".r")

            # plot landmark
            for i in range(calc_n_lm(xEst)):
                plt.plot(xEst[STATE_SIZE + i * 2],
                         xEst[STATE_SIZE + i * 2 + 1], "xg")

            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-b")
            plt.plot(hxDR[0, :],
                     hxDR[1, :], "-k")
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r")
            plt.axis("equal")
            plt.grid(True)
            if save_gif:
                fig.canvas.draw()
                frame = get_frame_as_array(fig)
                frames.append(frame)
            if show_animation: # and not save_gif:
                plt.pause(0.001)

    if save_gif and len(frames) > 0:
        imageio.mimsave(save_path, frames, fps=10, loop=0)  # 保存为GIF
        print(f"GIF saved to {save_path}")

if __name__ == '__main__':
    main()

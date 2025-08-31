import math

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
from utils.plot import get_frame_as_array
import imageio
save_gif = True


#  ICP parameters
EPS = 0.0001
MAX_ITER = 100

show_animation = True


def icp_matching(previous_points, current_points, frames):
    """
    Iterative Closest Point matching
    - input
    previous_points: 2D or 3D points in the previous frame
    current_points: 2D or 3D points in the current frame
    - output
    R: Rotation matrix
    T: Translation vector
    """
    H = None  # homogeneous transformation matrix

    dError = np.inf
    preError = np.inf
    count = 0
    residuals = []

    if show_animation:
        fig = plt.figure(figsize=(10,5))
        if previous_points.shape[0] == 3:
           fig.add_subplot(121, projection='3d')

    while dError >= EPS:
        count += 1

        # indexes, error = nearest_neighbor_association(previous_points, current_points)
        indexes, error = nearest_neighbor_association_cor(previous_points, current_points)
        Rt, Tt = svd_motion_estimation(previous_points[:, indexes], current_points)
        # update current points
        current_points = (Rt @ current_points) + Tt[:, np.newaxis]

        dError = preError - error
        residuals.append(error)
        print("Residual:", error)

        if show_animation:  # pragma: no cover
            plot_points(previous_points, current_points, fig, frames, residuals)
            plt.pause(0.1)

        if dError < 0:  # prevent matrix H changing, exit loop
            print("Not Converge...", preError, dError, count)
            break

        preError = error
        H = update_homogeneous_matrix(H, Rt, Tt)

        if dError <= EPS:
            print("Converge", error, dError, count)
            break
        elif MAX_ITER <= count:
            print("Not Converge...", error, dError, count)
            break

    R = np.array(H[0:-1, 0:-1])
    T = np.array(H[0:-1, -1])

    return R, T


def update_homogeneous_matrix(Hin, R, T):

    r_size = R.shape[0]
    H = np.zeros((r_size + 1, r_size + 1))

    H[0:r_size, 0:r_size] = R
    H[0:r_size, r_size] = T
    H[r_size, r_size] = 1.0

    if Hin is None:
        return H
    else:
        return Hin @ H


def nearest_neighbor_association(previous_points, current_points):

    # calc the sum of residual errors
    delta_points = previous_points - current_points
    d = np.linalg.norm(delta_points, axis=0)
    error = sum(d)

    # calc index with nearest neighbor assosiation
    d = np.linalg.norm(np.repeat(current_points, previous_points.shape[1], axis=1)
                       - np.tile(previous_points, (1, current_points.shape[1])), axis=0)
    indexes = np.argmin(d.reshape(current_points.shape[1], previous_points.shape[1]), axis=1)

    return indexes, error

# corrected
def nearest_neighbor_association_cor(previous_points, current_points):
    # Calculate all pairwise distances between current and previous points
    # Shape after repeat/tile: (dimension, N*M) where N=current points, M=previous points
    expanded_current = np.repeat(current_points, previous_points.shape[1], axis=1)
    expanded_previous = np.tile(previous_points, (1, current_points.shape[1]))
    d = np.linalg.norm(expanded_current - expanded_previous, axis=0)

    distance_matrix = d.reshape(current_points.shape[1], previous_points.shape[1])
    indexes = np.argmin(distance_matrix, axis=1)

    # Calculate error based on the actual nearest neighbor matches
    matched_previous = previous_points[:, indexes]  # Get matched previous points
    delta = matched_previous - current_points  # Difference between matched points
    error = np.sum(np.linalg.norm(delta, axis=0))  # Sum of L2 norms

    return indexes, error


# Kabsch/Umeyama 刚体配准
def svd_motion_estimation(previous_points, current_points):
    pm = np.mean(previous_points, axis=1)
    cm = np.mean(current_points, axis=1)

    p_shift = previous_points - pm[:, np.newaxis]
    c_shift = current_points - cm[:, np.newaxis]

    W = c_shift @ p_shift.T
    u, s, vh = np.linalg.svd(W)

    R = (u @ vh).T
    t = pm - (R @ cm)

    return R, t


def plot_points(previous_points, current_points, figure, frames, residuals):
    figure.clf()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    dim = previous_points.shape[0]

    if dim == 2:
        ax1 = figure.add_subplot(121)
        ax1.plot(previous_points[0, :], previous_points[1, :], ".r", label="Previous")
        ax1.plot(current_points[0, :], current_points[1, :], ".b", label="Current")
        ax1.plot(0.0, 0.0, "xr")
        ax1.axis("equal")
        ax1.legend()

    elif dim == 3:
        ax1 = figure.add_subplot(121, projection='3d')
        ax1.scatter(previous_points[0, :], previous_points[1, :], previous_points[2, :], c="r", marker=".",
                   label="Previous")
        ax1.scatter(current_points[0, :], current_points[1, :], current_points[2, :], c="b", marker=".", label="Current")
        ax1.scatter(0.0, 0.0, 0.0, c="k", marker="x")
        ax1.legend()

    ax2 = figure.add_subplot(122)
    ax2.plot(residuals, "-r")
    ax2.set_title("Residual Error")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error")

    figure.canvas.draw()
    if save_gif:
        frame = get_frame_as_array(figure)
        frames.append(frame)


def convert_rt_to_motion(R, T, dim=2):
    """
    将ICP求解的旋转矩阵R和平移向量T转换为与预设motion一致的形式
    :param R: 旋转矩阵 (2x2 或 3x3)
    :param T: 平移向量 (2x1 或 3x1)
    :param dim: 维度 (2或3)
    :return: 转换后的motion列表 [x, y, yaw] (2D) 或 [x, y, z, roll] (3D)
    """
    if dim == 2:
        # 2D情况下：从旋转矩阵提取yaw角（注意符号，因为ICP求解的是逆变换）
        yaw = -np.arctan2(R[1, 0], R[0, 0])  # 负号补偿逆变换
        # 平移向量直接取负（ICP求解的是从current到previous的变换，与motion方向相反）
        x = -T[0]
        y = -T[1]
        return [x, y, np.rad2deg(yaw)]
    elif dim == 3:
        # 3D情况下（绕x轴旋转）：从旋转矩阵提取roll角
        roll = -np.arcsin(-R[1, 2])  # 负号补偿逆变换
        # 平移向量取负
        x = -T[0]
        y = -T[1]
        z = -T[2]
        return [x, y, z, np.rad2deg(roll)]
    else:
        raise ValueError("维度必须为2或3")

def main():
    """
    主函数，执行点云配准仿真
    该函数生成随机点云，应用运动变换，并使用ICP算法进行点云配准。
    最后可选择将配准过程保存为GIF动画。
    """
    print(__file__ + " start!!")

    # simulation parameters
    nPoint = 1000
    fieldLength = 50.0
    motion = [0.5, 2.0, np.deg2rad(-10.0)]  # movement [x[m],y[m],yaw[deg]]

    nsim = 1  # number of simulation

    frames = []

    for _ in range(nsim):

        # previous points
        px = (np.random.rand(nPoint) - 0.5) * fieldLength
        py = (np.random.rand(nPoint) - 0.5) * fieldLength
        previous_points = np.vstack((px, py))

        # current points
        cx = [math.cos(motion[2]) * x - math.sin(motion[2]) * y + motion[0]
              for (x, y) in zip(px, py)]
        cy = [math.sin(motion[2]) * x + math.cos(motion[2]) * y + motion[1]
              for (x, y) in zip(px, py)]
        current_points = np.vstack((cx, cy))

        R, T = icp_matching(previous_points, current_points, frames)
        print("R:", R)
        print("T:", T)
        est_motion = convert_rt_to_motion(R, T, 2)
        print("est_motion:", est_motion)
        if save_gif:
            imageio.mimsave("icp_2d.gif", frames, fps=10, loop=1)

def main_3d_points():
    print(__file__ + " start!!")

    # simulation parameters for 3d point set
    nPoint = 1000
    fieldLength = 50.0
    motion = [0.5, 2.0, -5, np.deg2rad(-10.0)]  # [x[m],y[m],z[m],roll[deg]]

    nsim = 1  # number of simulation
    frames = []
    for _ in range(nsim):

        # previous points
        px = (np.random.rand(nPoint) - 0.5) * fieldLength
        py = (np.random.rand(nPoint) - 0.5) * fieldLength
        pz = (np.random.rand(nPoint) - 0.5) * fieldLength
        previous_points = np.vstack((px, py, pz))

        # current points
        cx = [math.cos(motion[3]) * x - math.sin(motion[3]) * z + motion[0]
              for (x, z) in zip(px, pz)]
        cy = [y + motion[1] for y in py]
        cz = [math.sin(motion[3]) * x + math.cos(motion[3]) * z + motion[2]
              for (x, z) in zip(px, pz)]
        current_points = np.vstack((cx, cy, cz))

        R, T = icp_matching(previous_points, current_points, frames)
        print("R:", R)
        print("T:", T)
        est_motion = convert_rt_to_motion(R, T, 3)
        print("est_motion:", est_motion)
        if save_gif:
            imageio.mimsave("icp_3d.gif", frames, fps=10, loop=1)

if __name__ == '__main__':
    main()
    #main_3d_points()

import open3d as o3d
import numpy as np
import glob
from colorsys import hsv_to_rgb


def load_bin(velo_filename):
    """从bin文件加载点云数据"""
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))  # 假设点云数据格式为(x, y, z, intensity)
    return scan


def get_distinct_colors(n):
    """生成n种不同的颜色，在HSV色彩空间均匀分布"""
    colors = []
    for i in range(n):
        # HSV颜色空间，H值从0到1变化，确保颜色差异明显
        hue = i / n
        saturation = 0.7
        value = 0.9
        # 转换为RGB
        r, g, b = hsv_to_rgb(hue, saturation, value)
        colors.append((r, g, b))
    return colors


def create_point_cloud(points, color=None):
    """将numpy数组转换为Open3D点云格式，可指定颜色"""
    pcd = o3d.geometry.PointCloud()
    # 只取前三个坐标值(x, y, z)
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # 设置点云颜色
    if color is not None:
        # 为所有点设置相同的颜色
        colors = np.tile(color, (points.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # 如果未指定颜色，使用强度值作为颜色
        if points.shape[1] >= 4:
            # 归一化强度值到0-1范围
            intensities = points[:, 3]
            intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-10)
            # 创建RGB颜色（这里使用强度作为灰度值）
            colors = np.zeros((points.shape[0], 3))
            colors[:, :] = intensities[:, np.newaxis]
            pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_point_clouds(point_clouds, window_name="点云可视化"):
    """可视化点云列表"""
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)

    # 添加所有点云
    for pcd in point_clouds:
        vis.add_geometry(pcd)

    # 设置视角和渲染选项
    opt = vis.get_render_option()
    opt.background_color = [0.0, 0.0, 0.0]  # 黑色背景
    opt.point_size = 1.0  # 点大小

    # 运行可视化
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    # 获取所有bin文件路径
    bin_files = glob.glob('./*.bin')

    if not bin_files:
        print("未找到任何.bin文件")
    else:
        num_files = len(bin_files)
        print(f"找到{num_files}个bin文件，正在加载...")

        # 生成与文件数量相同的不同颜色
        frame_colors = get_distinct_colors(num_files)

        # 加载并转换所有点云
        point_clouds = []
        for i, file_path in enumerate(bin_files):
            print(f"处理文件: {file_path}")
            # 从bin文件加载点云
            bin_points = load_bin(file_path)
            # 转换为Open3D格式，并使用第i种颜色
            pcd = create_point_cloud(bin_points, color=frame_colors[i])
            point_clouds.append(pcd)

            # 可视化点云
            visualize_point_clouds(point_clouds)

import matplotlib.pyplot as plt
from maze_generator import MazeGenerator
from dijkstra import DijkstraPlanner
from matplotlib.colors import ListedColormap

def plan_path_in_maze(width, height, grid_size, robot_radius):
    # 生成迷宫
    mg = MazeGenerator(width, height)
    maze = mg.generate()
    entry_x, entry_y = mg.entry  # 入口
    exit_x, exit_y = mg.exit  # 出口

    # 将迷宫转换为 Dijkstra 算法所需的障碍物列表
    ox, oy = [], []
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y][x] == 1:
                ox.append(x)
                oy.append(y)

    # 创建 Dijkstra 规划器
    dijkstra = DijkstraPlanner(ox, oy, grid_size, robot_radius)

    # 调整起点和终点，避免在最外侧
    if entry_x == 0:
        entry_x += 1
    elif entry_x == width - 1:
        entry_x -= 1
    if entry_y == 0:
        entry_y += 1
    elif entry_y == height - 1:
        entry_y -= 1

    if exit_x == 0:
        exit_x += 1
    elif exit_x == width - 1:
        exit_x -= 1
    if exit_y == 0:
        exit_y += 1
    elif exit_y == height - 1:
        exit_y -= 1

    # 进行路径规划
    rx, ry = dijkstra.planning(entry_x, entry_y, exit_x, exit_y)

    # 可视化迷宫和路径
    cmap = ListedColormap(['white', 'black'])  # plt.colormaps.get_cmap('gray')
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap=cmap)
    plt.gca().invert_yaxis()  # 适配坐标原点在左下
    plt.plot(entry_x, entry_y, 'go', markersize=10)
    plt.plot(exit_x, exit_y, 'ro', markersize=10)
    plt.plot(rx, ry, '-b', linewidth=2)
    plt.title(f'Path in Maze (Entry: {mg.entry}, Exit: {mg.exit})')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return rx, ry

if __name__ == "__main__":
    width = 51
    height = 51
    grid_size = 1.0
    robot_radius = 0.9  # 小于grid_size
    plan_path_in_maze(width, height, grid_size, robot_radius)
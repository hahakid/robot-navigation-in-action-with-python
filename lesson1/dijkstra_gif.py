"""
Grid based Dijkstra planning with optimized GIF output
"""

import matplotlib.pyplot as plt
import math
import imageio
from utils.plot import get_frame_as_array

show_animation = True
save_gif = True  # 是否保存GIF
output_path = "dijkstra_animation.gif"  # GIF输出路径
gridmap_save_interval = 100  # 栅格地图保存间隔


class DijkstraPlanner:
    def __init__(self, ox, oy, resolution, robot_radius):
        """
        Initialize map for Dijkstra planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_width = None
        self.y_width = None
        self.obstacle_map = None

        self.resolution = resolution
        self.robot_radius = robot_radius
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index  # index of previous Node

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy, fig=None, ax=None, frames=None, gridmap_interval=10):
        """
        Dijkstra path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start_node)] = start_node

        iteration_count = 0  # 迭代计数器

        while True:
            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            # show graph
            if show_animation or (fig and ax):
                if ax:
                    ax.plot(self.calc_position(current.x, self.min_x),
                            self.calc_position(current.y, self.min_y), "xc")
                else:
                    plt.plot(self.calc_position(current.x, self.min_x),
                             self.calc_position(current.y, self.min_y), "xc")

                # 保存帧：栅格地图阶段按间隔保存，路径生成阶段每帧保存
                save_this_frame = False
                if frames is not None:
                    if len(closed_set) < len(open_set) + len(closed_set) * 0.5:
                        # 栅格地图阶段（搜索早期）：按间隔保存
                        save_this_frame = (iteration_count % gridmap_interval == 0)
                        print(iteration_count, save_this_frame)
                    else:
                        # 路径生成阶段（搜索后期）：每帧保存
                        save_this_frame = True

                    if save_this_frame:
                        ax.set_title(f"Dijkstra Path Planning - Nodes: {len(closed_set)}")
                        fig.canvas.draw()
                        frames.append(get_frame_as_array(fig))
                elif show_animation:
                    plt.pause(0.001)

            iteration_count += 1

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand search grid based on motion model
            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_index(node)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # Discover a new node
                else:
                    if open_set[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_position(goal_node.x, self.min_x)], [
            self.calc_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    def calc_position(self, index, minp):
        pos = index * self.resolution + minp
        return pos

    def calc_xy_index(self, position, minp):
        return round((position - minp) / self.resolution)

    def calc_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        if py < self.min_y:
            return False
        if px >= self.max_x:
            return False
        if py >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("Map bounds:")
        print(f"  X: {self.min_x} to {self.max_x}")
        print(f"  Y: {self.min_y} to {self.max_y}")

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print(f"Grid size: {self.x_width} x {self.y_width}")

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = -5.0  # [m]
    sy = -5.0  # [m]
    gx = 30.0  # [m]
    gy = 30.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 40):  # 底边， (-10,-10)到(40,-10)
        ox.append(float(i))
        oy.append(-10.0)
    for i in range(-10, 40):
        ox.append(40.0)
        oy.append(float(i))
    for i in range(-10, 41):
        ox.append(float(i))
        oy.append(40.0)
    for i in range(-10, 41):
        ox.append(-10.0)
        oy.append(float(i))
    for i in range(-10, 20):  # 左侧障碍墙， 由下到上，不封顶
        ox.append(5.0)
        oy.append(float(i))
    for i in range(0, 20):  # 右侧障碍墙，由上到下，不封顶
        ox.append(25.0)
        oy.append(40.0 - i)

    # 初始化图形
    if show_animation or save_gif:
        if save_gif:
            fig = plt.figure(figsize=(10, 8), dpi=100)
            ax = fig.add_subplot(111)
            frames = []
        else:
            fig, ax = plt.figure(), plt.gca()

        ax.plot(ox, oy, ".k", markersize=5)
        ax.plot(sx, sy, "og", markersize=10)
        ax.plot(gx, gy, "xb", markersize=10)
        ax.grid(True)
        ax.set_title("Dijkstra Path Planning")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

        if save_gif:
            fig.canvas.draw()
            frames.append(get_frame_as_array(fig))
        elif show_animation:
            plt.show(block=False)

    # 执行规划，传入栅格地图保存间隔
    dijkstra = DijkstraPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = dijkstra.planning(
        sx, sy, gx, gy,
        fig, ax,
        frames if save_gif else None,
        gridmap_save_interval
    )

    # 绘制最终路径
    if show_animation or save_gif:
        ax.plot(rx, ry, "-r", linewidth=3)

        if save_gif:
            ax.set_title("Dijkstra Path Planning - Path Found!")
            fig.canvas.draw()
            for i in range(15):  # 重复绘制，持续显示
                frames.append(get_frame_as_array(fig))

            # 保存GIF
            imageio.mimsave(output_path, frames, fps=15, loop=1)
            print(f"GIF动画已保存至: {output_path}")
        elif show_animation:
            plt.draw()
            plt.pause(0.01)
            plt.show()


if __name__ == '__main__':
    main()
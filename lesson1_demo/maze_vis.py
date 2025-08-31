import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

import imageio
from utils.plot import get_frame_as_array


class MazeGenerator:
    def __init__(self, width, height):
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height # + 1
        self.maze = np.ones((self.height, self.width), dtype=int)  # 1=墙，0=通路
        self.visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.generation_steps = []  # 存储生成过程的每一步


    def generate(self):
        # 初始化起点
        start_x, start_y = 1, 1
        self.maze[start_y][start_x] = 0  # 起点设为通路
        self.visited[start_y][start_x] = True
        self.generation_steps.append(np.copy(self.maze))  # 记录初始状态

        stack = [(start_x, start_y)]
        directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]  # 移动单位为2（中间留墙）

        while stack:
            x, y = stack[-1]  # 当前位置（栈顶）
            unvisited_neighbors = []

            # 寻找未访问的邻居
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # 检查邻居是否在迷宫范围内且未访问
                if 0 < nx < self.width - 1 and 0 < ny < self.height - 1 and not self.visited[ny][nx]:
                    unvisited_neighbors.append((dx, dy))

            if unvisited_neighbors:
                # 随机选择一个邻居，打通墙壁
                dx, dy = random.choice(unvisited_neighbors)
                # 计算中间墙的位置
                wall_x, wall_y = x + dx // 2, y + dy // 2
                self.maze[wall_y][wall_x] = 0  # 打通墙
                # 标记新位置为通路
                new_x, new_y = x + dx, y + dy
                self.maze[new_y][new_x] = 0
                self.visited[new_y][new_x] = True
                stack.append((new_x, new_y))
                # 记录当前步骤
                self.generation_steps.append(np.copy(self.maze))
            else:
                # 回溯（无未访问邻居时出栈）
                stack.pop()
                # 回溯步骤也记录（可选，展示回溯过程）
                self.generation_steps.append(np.copy(self.maze))

        # 设置入口和出口
        self._set_entry_exit()
        self.generation_steps.append(np.copy(self.maze))  # 记录最终状态
        return self.maze

    def _set_entry_exit(self):
        sides = ['top', 'right', 'bottom', 'left']
        entry_side, exit_side = random.sample(sides, 2)

        # 设置入口
        if entry_side == 'top':
            entry_y, entry_x = 0, random.randrange(1, self.width-1, 2)
        elif entry_side == 'right':
            entry_y, entry_x = random.randrange(1, self.height-1, 2), self.width-1
        elif entry_side == 'bottom':
            entry_y, entry_x = self.height-1, random.randrange(1, self.width-1, 2)
        else:
            entry_y, entry_x = random.randrange(1, self.height-1, 2), 0

        # 设置出口
        if exit_side == 'top':
            exit_y, exit_x = 0, random.randrange(1, self.width-1, 2)
            while exit_x == entry_x: exit_x = random.randrange(1, self.width-1, 2)
        elif exit_side == 'right':
            exit_y, exit_x = random.randrange(1, self.height-1, 2), self.width-1
            while exit_y == entry_y: exit_y = random.randrange(1, self.height-1, 2)
        elif exit_side == 'bottom':
            exit_y, exit_x = self.height-1, random.randrange(1, self.width-1, 2)
            while exit_x == entry_x: exit_x = random.randrange(1, self.width-1, 2)
        else:
            exit_y, exit_x = random.randrange(1, self.height-1, 2), 0
            while exit_y == entry_y: exit_y = random.randrange(1, self.height-1, 2)

        self.maze[entry_y][entry_x] = 0
        self.maze[exit_y][exit_x] = 0
        self.entry, self.exit = (entry_x, entry_y), (exit_x, exit_y)

    def visualize_generation(self, interval=100):
        """可视化生成步骤（动画形式）"""
        cmap = ListedColormap(['white', 'black'])  # 墙=黑色，通路=白色
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.matshow(self.generation_steps[0], cmap=cmap)
        ax.set_title('Maze Generation Process (DFS)')
        ax.set_xticks([])
        ax.set_yticks([])

        # 更新动画帧
        def update(frame):
            im.set_data(self.generation_steps[frame])
            return im,

        # 创建动画
        anim = FuncAnimation(
            fig, update, frames=len(self.generation_steps),
            interval=interval, blit=True
        )
        plt.show()
        return anim

    def save_generation_gif(self, filename="maze_generation.gif", interval=100, fps=10):
        """将生成过程保存为GIF"""
        cmap = ListedColormap(['white', 'black','red', 'blue'])  # 墙=黑色，通路=白色
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.matshow(self.generation_steps[0], cmap=cmap)

        ax.set_title('Maze Generation (DFS)')
        ax.set_xticks([])
        ax.set_yticks([])

        # 用于存储每一帧的图像
        frames = []

        # 更新动画帧并保存
        def update(frame):
            im.set_data(self.generation_steps[frame])

            # 将当前帧转换为图像数组
            fig.canvas.draw()
            frame_array = get_frame_as_array(fig)
            frames.append(frame_array)
            print(f"{frame} added.")
            return im,

        # 创建动画（不显示，仅生成帧）
        anim = FuncAnimation(
            fig, update, frames=len(self.generation_steps),
            interval=interval, blit=True
        )
        plt.show()  # close(fig)  # 关闭窗口，避免显示

        # 保存为GIF
        imageio.mimsave(filename, frames, fps=fps, loop=0)

        print(f"GIF已保存为: {filename}")
        #plt.show()
        return anim

if __name__ == "__main__":
    # 生成一个小型迷宫（便于快速展示步骤）
    mg = MazeGenerator(21, 21)  # 21x21的迷宫
    maze = mg.generate()
    # 可视化生成步骤（interval为每帧间隔毫秒）

    # mg.visualize_generation(interval=50)
    mg.save_generation_gif("maze_generation.gif", interval=50, fps=15)
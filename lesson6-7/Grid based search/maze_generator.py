import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class MazeGenerator:
    def __init__(self, width, height):
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.maze = np.ones((self.height, self.width), dtype=int)
        self.visited = [[False for _ in range(self.width)] for _ in range(self.height)]

    def generate(self):
        start_x, start_y = 1, 1
        self.maze[start_y][start_x] = 0
        self.visited[start_y][start_x] = True
        stack = [(start_x, start_y)]
        directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]

        while stack:
            x, y = stack[-1]
            unvisited_neighbors = []

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.width - 1 and 0 < ny < self.height - 1 and not self.visited[ny][nx]:
                    unvisited_neighbors.append((dx, dy))

            if unvisited_neighbors:
                dx, dy = random.choice(unvisited_neighbors)
                wall_x, wall_y = x + dx // 2, y + dy // 2
                self.maze[wall_y][wall_x] = 0
                new_x, new_y = x + dx, y + dy
                self.maze[new_y][new_x] = 0
                self.visited[new_y][new_x] = True
                stack.append((new_x, new_y))
            else:
                stack.pop()

        self._set_entry_exit()
        return self.maze

    def _set_entry_exit(self):
        sides = ['top', 'right', 'bottom', 'left']
        entry_side, exit_side = random.sample(sides, 2)

        if entry_side == 'top':
            entry_y = 0
            entry_x = random.randrange(1, self.width - 1, 2)
        elif entry_side == 'right':
            entry_y = random.randrange(1, self.height - 1, 2)
            entry_x = self.width - 1
        elif entry_side == 'bottom':
            entry_y = self.height - 1
            entry_x = random.randrange(1, self.width - 1, 2)
        else:
            entry_y = random.randrange(1, self.height - 1, 2)
            entry_x = 0

        if exit_side == 'top':
            exit_y = 0
            exit_x = random.randrange(1, self.width - 1, 2)
            while exit_x == entry_x:
                exit_x = random.randrange(1, self.width - 1, 2)
        elif exit_side == 'right':
            exit_y = random.randrange(1, self.height - 1, 2)
            exit_x = self.width - 1
            while exit_y == entry_y:
                exit_y = random.randrange(1, self.height - 1, 2)
        elif exit_side == 'bottom':
            exit_y = self.height - 1
            exit_x = random.randrange(1, self.width - 1, 2)
            while exit_x == entry_x:
                exit_x = random.randrange(1, self.width - 1, 2)
        else:
            exit_y = random.randrange(1, self.height - 1, 2)
            exit_x = 0
            while exit_y == entry_y:
                exit_y = random.randrange(1, self.height - 1, 2)

        self.maze[entry_y][entry_x] = 0
        self.maze[exit_y][exit_x] = 0
        self.entry = (entry_x, entry_y)
        self.exit = (exit_x, exit_y)

    def visualize(self):
        # 统一使用反色配置：路径白色，墙壁黑色
        cmap = ListedColormap(['white', 'black', 'green', 'red'])

        plt.figure(figsize=(10, 10))

        vis_data = self.maze.copy()
        vis_data[self.entry[1], self.entry[0]] = 2
        vis_data[self.exit[1], self.exit[0]] = 3

        plt.matshow(vis_data, cmap=cmap, fignum=1)
        plt.gca().invert_yaxis()  # 修改坐标轴方向，使原点在左下角
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Random Maze (Entry: {self.entry}, Exit: {self.exit})')
        plt.show()


if __name__ == "__main__":
    mg = MazeGenerator(21, 25)
    maze = mg.generate()
    mg.visualize()  # 不再需要参数
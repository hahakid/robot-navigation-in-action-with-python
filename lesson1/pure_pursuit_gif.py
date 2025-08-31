import matplotlib.pyplot as plt
import imageio
import numpy as np

# 1. 初始化图形（固定尺寸和参数）
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_xlim(0, 10)  # 固定坐标轴范围
ax.set_ylim(0, 10)
plt.ion()  # 开启交互模式

frames = []  # 存储帧的列表

# 2. 循环生成帧
for i in range(50):
    # 更新画面内容（例如绘制一个移动的点）
    ax.cla()
    ax.plot(i/5, np.sin(i/5), 'ro')  # 动态内容
    ax.set_title(f"Frame {i}")
    ax.grid(True)
    
    # 3. 同步渲染、显示和保存
    fig.canvas.draw()  # 强制渲染当前帧
    
    # 显示：刷新交互界面
    plt.pause(0.1)  # 控制显示帧率
    
    # 保存：捕获当前渲染的帧（与显示内容完全一致）
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame)

# 4. 最终生成GIF
imageio.mimsave("output.gif", frames, fps=10)
plt.ioff()  # 关闭交互模式
plt.show()  # 显示最终结果
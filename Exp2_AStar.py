import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# 验证颜色名称是否合法
valid_colors = list(mcolors.CSS4_COLORS.keys())  # CSS4中有效的颜色名称

# 定义网格大小
nrows, ncols = 10, 10

# 创建颜色矩阵，默认为白色
color_matrix = np.full((nrows, ncols), 'white')

# 设置特定的单元格颜色
# 使用确保是有效的颜色名称
specified_colors = {
    (1, 1): 'yellow',  # 第2行第2列设置为黄色
    (3, 4): 'blue',  # 第4行第5列设置为蓝色
    (7, 7): 'yellow',  # 第8行第8列设置为黄色
    (8, 2): 'blue'  # 第9行第3列设置为蓝色
}

# 验证所有指定的颜色名称都是有效的
for pos, color in specified_colors.items():
    if color not in valid_colors:
        raise ValueError(f"Invalid color specified: {color}")

# 更新颜色矩阵
for pos, color in specified_colors.items():
    color_matrix[pos] = color  # 设置特定单元格的颜色

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制网格图
for row in range(nrows):
    for col in range(ncols):
        color = color_matrix[row, col]  # 从颜色矩阵中获取颜色
        ax.add_patch(patches.Rectangle((col, nrows - row - 1), 1, 1, color=color))  # 绘制矩形

# 设置坐标轴
ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)
ax.set_xticks(np.arange(ncols + 1))
ax.set_yticks(np.arange(nrows + 1))
ax.grid(True)

plt.show()
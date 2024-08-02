import matplotlib.pyplot as plt
import numpy as np

# 数据
data = {
    'Noise factor': [0, 0.1, 0.2, 0.3 ,0.4],
    'MDaDL': [94.86,87.38,88.91,83.59,79.36],
    'ANML': [94.79,90.77,86.01,80.01,75.91],
    'KGMML': [95.07,90.25,84.17,80.35,73.03],
    'psub': [93.16,91.39,86.11,82.11,79.17],
    'BMLFSP': [93.66,91.99,82.69,78.29,76.18],
    'DLMPM': [95.17,93.16,89.63,84.84,81.28],
}

# 绘制图表
fig, ax = plt.subplots()
ax.plot(data['Noise factor'], data['MDaDL'], label='MDaDL', linestyle='-', color='red', marker='o', markersize=5, linewidth=2, alpha=0.7)
ax.plot(data['Noise factor'], data['ANML'], label='ANML', linestyle='--', color='yellow', marker='v', markersize=5, linewidth=2, alpha=0.7)
ax.plot(data['Noise factor'], data['KGMML'], label='KGMML', linestyle='-.', color='green', marker='^', markersize=5, linewidth=2, alpha=0.7)
ax.plot(data['Noise factor'], data['psub'], label='psub', linestyle=':', color='blue', marker='s', markersize=5, linewidth=2, alpha=0.7)
ax.plot(data['Noise factor'], data['BMLFSP'], label='BMLFSP', linestyle='-', color='purple', marker='*', markersize=5, linewidth=2, alpha=0.7)
ax.plot(data['Noise factor'], data['DLMPM'], label='DLMPM', linestyle='--', color='orange', marker='p', markersize=5, linewidth=2, alpha=0.7)

# 添加标题和标签
ax.set_title('Breast Cancer')
ax.set_xlabel('Noise factor')
ax.set_ylabel('Accuracy')

# 添加图例
ax.legend(loc='upper right', shadow=True)


# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('Cancer01.png')
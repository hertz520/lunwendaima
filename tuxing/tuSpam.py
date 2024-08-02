import matplotlib.pyplot as plt
import numpy as np

# 数据
data = {
    'Noise factor': [0, 0.1, 0.2, 0.3 ,0.4],
    'MDaDL': [87.45,87.38,85.91,84.59,80.31],
    'ANML': [85.91,84.77,84.01,82.01,81.09],
    'KGMML': [86.15,87.25,84.17,80.35,79.03],
    'psub': [84.92,84.39,82.11,81.11,80.95],
    'BMLFSP': [84.91,80.99,78.69,74.29,74.11],
    'DLMPM': [89.56,88.16,85.63,84.84,79.11],
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
ax.set_title('Spam')
ax.set_xlabel('Noise factor')
ax.set_ylabel('Accuracy')

# 添加图例
ax.legend(loc='upper right', shadow=True)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('Spam01.png')
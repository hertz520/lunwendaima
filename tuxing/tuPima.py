import matplotlib.pyplot as plt
import numpy as np

# 数据
data = {
    'Noise factor': [0, 0.1, 0.2, 0.3 ,0.4],
    'MDaDL': [70.13,70.38,69.91,66.59,65.97],
    'ANML': [64.83,64.77,63.01,60.01,59.39],
    'KGMML': [72.37,70.65,70.87,65.35,64.81],
    'psub': [61.06,61.39,58.11,57.11,56.98],
    'BMLFSP': [60.94,59.99,58.69,57.29,56.79],
    'DLMPM': [72.58,72.16,71.63,68.84,67.71],
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
ax.set_title('Pima')
ax.set_xlabel('Noise factor')
ax.set_ylabel('Accuracy')

# 添加图例
ax.legend(loc='upper right', shadow=True)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('Pima01.png')
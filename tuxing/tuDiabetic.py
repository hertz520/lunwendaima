import matplotlib.pyplot as plt
import numpy as np

# 数据
data = {
    'Noise factor': [0, 0.1, 0.2, 0.3 ,0.4],
    'MDaDL': [68.36,66.38,64.91,64.39,61.58],
    'ANML': [60.05,57.77,56.01,55.62,54.15],
    'KGMML': [67.61,64.65,60.87,59.35,56.92],
    'psub': [65.83,64.39,64.11,63.65,61.92],
    'BMLFSP': [60.19,57.99,55.69,54.29,52.92],
    'DLMPM': [69.82,67.96,66.63,64.84,64.11],
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
ax.set_title('Diabetic')
ax.set_xlabel('Noise factor')
ax.set_ylabel('Accuracy')

# 添加图例
ax.legend(loc='upper right', shadow=True)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('Diabetic01.png')
import matplotlib.pyplot as plt
import numpy as np

# 提供的数据点
tpr = [0.833, 0.9286, 1, 0.9125, 0.947, 0.818, 1, 0.933, 1, 1]
fpr = [0.0314, 0.055, 0.0424, 0.00317, 0.033, 0.154, 0.2, 0.102, 0.0018, 0]

# 创建一个标签列表
labels = [f"Point {i+1}" for i in range(len(tpr))]

x = np.arange(len(labels))  # 标签位置
width = 0.35  # 柱的宽度

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, tpr, width, label='TPR')
rects2 = ax.bar(x + width/2, fpr, width, label='FPR')

# 添加文本标签、标题和自定义X轴刻度等
ax.set_xlabel('Data Points')
ax.set_ylabel('Rates')
ax.set_title('TPR and FPR for Different Points(optimal thresholds)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()


'''
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, marker='o', linestyle='-', color='b')
plt.plot([0, 1], [0, 1], 'r--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True)
plt.show()
'''
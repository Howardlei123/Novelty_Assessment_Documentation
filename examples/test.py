import subprocess
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import shutil

def align_labels(predicted_labels, true_labels, window_size=20):
    """
    Adjusts the true labels to align with the predicted labels if the predicted labels are delayed
    but correct within a specified window size.
    """
    visited = [0] * len(predicted_labels) # Record previously modified true_labels to prevent double mapping
    paired = [0] * len(predicted_labels) # Record originally matching signals to prevent random alterations
    # Ensure the lists are of equal length
    if len(predicted_labels) != len(true_labels):
        raise ValueError("The length of predicted_labels and true_labels must be the same.")

    adjusted_true_labels = true_labels.copy() 
    
    # Iterate over the predicted labels
    for i, pred_label in enumerate(predicted_labels):
        if pred_label == 1 and true_labels[i] == 1:
            paired[i] = 1;
        elif pred_label == 1 and true_labels[i] == 0:
            # Check for a matching true label within the window size
            for j in range(max(0, i - window_size), i):
                if true_labels[j] == 1 and visited[j] == 0 and paired[j] == 0:
                    visited[j] = 1
                    # Adjust the true label to align with the predicted label
                    adjusted_true_labels[j] = 0
                    adjusted_true_labels[i] = 1
                    break

    return adjusted_true_labels

df_t = pd.read_csv(r'truelabel\truelabel_dataset5.csv')

label_t = list(df_t['label'])

df_p = pd.read_csv(r'predictedlabel.csv')

label_p = list(df_p['label'])

tl = align_labels(label_p,label_t,20)

tn, fp, fn, tp = confusion_matrix(tl, label_p).ravel()

# 计算指标
precision = precision_score(tl, label_p)  # 精确度
recall = recall_score(tl, label_p)  # 召回率
fpr = fp / (fp + tn)  # 假阳性率
tpr = tp / (tp + fn)  # 真阳性率

# 打印结果
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"False Positive Rate: {fpr}")
print(f"True Positive Rate: {tpr}")

fpr_array, tpr_array, _ = roc_curve(tl, label_p)  # 获取ROC曲线
roc_auc = roc_auc_score(tl, label_p)  # 计算AUC

plt.figure()
plt.plot(fpr_array, tpr_array, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05]) 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#把这些加进去，并且自动记录到表格里面
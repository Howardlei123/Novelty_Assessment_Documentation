import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

df = pd.read_excel('results/results_Dataset3.xlsx', engine='openpyxl')  # change to the directory of true labels

tpr = df['TPR'].tolist()
fpr = df['FPR'].tolist()

fpr = [0] + fpr + [1]
tpr = [0] + tpr + [1]

inds = np.argsort(fpr)
sorted_fpr = np.array(fpr)[inds]
sorted_tpr = np.array(tpr)[inds]

roc_auc = auc(sorted_fpr, sorted_tpr)

plt.figure()
lw = 2  
plt.plot(sorted_fpr, sorted_tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

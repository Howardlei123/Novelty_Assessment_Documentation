import subprocess
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


base_path = os.path.join(os.getcwd(), 'examples')
file_name = "track.py"
script_path = os.path.join(base_path, file_name)
roc_directory = os.path.join(base_path, r'roc')
python_interpreter = r"C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe" # address to the python that that dependancies are installed.

max_window_size = 20 #initial

#thresholds

df_t = pd.read_csv(r'examples\truelabel\truelabel_dataset3.csv')
label_t = list(df_t['label'])


def align_labels(predicted_labels, true_labels, max_window_size,frame_differences):
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
            for j in range(max(0, i - max_window_size), i):
                if true_labels[j] == 1 and visited[j] == 0 and paired[j] == 0:
                    visited[j] = 1;
                    # Adjust the true label to align with the predicted label
                    adjusted_true_labels[j] = 0
                    adjusted_true_labels[i] = 1
                    frame_differences.append(abs(i - j))
                    break


command = [python_interpreter, script_path]
        
subprocess.run(command)

df_p = pd.read_csv(r'examples\predictedlabel.csv')
label_p = list(df_p['label'])

frame_differences = []

align_labels(label_p, label_t, max_window_size,frame_differences)

plt.hist(frame_differences, bins=range(max(frame_differences)+1), edgecolor='black', align='left')
plt.title('Histogram of Frame Differences between Matched Items')
plt.xlabel('Frame Difference')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

plt.savefig(os.path.join(roc_directory, 'Frame_Differences_Histogram.png'))
plt.close()
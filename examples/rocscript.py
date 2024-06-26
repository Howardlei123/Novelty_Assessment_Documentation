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
from pathlib import Path

base_path = os.path.join(os.getcwd(), 'examples')
file_name = "track.py"
script_path = os.path.join(base_path, file_name)
roc_directory = os.path.join(base_path, r'roc')
python_interpreter = r"C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe" # address to the python that that dependancies are installed.
window_size = 9

# need to change
df_t = pd.read_csv(r'examples\truelabel\truelabel_dataset3.csv')
excel_directory = Path.cwd() / 'results' 
if not os.path.exists(excel_directory): 
        os.makedirs(excel_directory)
excel_file = os.path.join(excel_directory, 'results_Dataset3.xlsx')#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

label_t = list(df_t['label'])

#delete all the graphs in roc file 
if os.path.exists(roc_directory):
    for filename in os.listdir(roc_directory):
        file_path = os.path.join(roc_directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

#thresholds name

param1_name = "Movement Threshold"
param2_name = "Diff Threshold"
param3_name = "ROI Threshold"

#thresholds value dimension must be the same
movement_threshold_values = [0,30,50]   
diff_threshold_values = [0,50,100] 
roi_threshold_values = [0,150,300] 
AUC_values = np.zeros((len(movement_threshold_values), len(diff_threshold_values), len(roi_threshold_values)))


fpr_matrix = np.zeros((len(movement_threshold_values), len(diff_threshold_values), len(roi_threshold_values)))
fnr_matrix = np.zeros((len(movement_threshold_values), len(diff_threshold_values), len(roi_threshold_values)))
tpr_matrix = np.zeros((len(movement_threshold_values), len(diff_threshold_values), len(roi_threshold_values)))
results = pd.DataFrame()

def align_labels(predicted_labels, true_labels, window_size):
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

rows_list = []

for i, param1 in enumerate(movement_threshold_values):
    for j, param2 in enumerate(diff_threshold_values):
        for k, param3 in enumerate(roi_threshold_values):

            custom_args = [
                '--movement-threshold', str(param1), 
                '--diff-threshold', str(param2),
                '--roi-threshold',str(param3)
            ]

            command = [python_interpreter, script_path] + custom_args
        
            subprocess.run(command)
   
            df_p = pd.read_csv(r'examples\predictedlabel.csv')
            label_p = list(df_p['label'])


            adjusted_true_labels = align_labels(label_p, label_t, window_size)

            tn, fp, fn, tp = confusion_matrix(adjusted_true_labels, label_p).ravel()

            precision = precision_score(adjusted_true_labels, label_p)  # 精确度
            recall = recall_score(adjusted_true_labels, label_p)  # 召回率

            fpr, tpr, _ = roc_curve(adjusted_true_labels, label_p)
            
            roc_auc = auc(fpr, tpr)

            AUC_values[i, j, k] = roc_auc
            
             # ROC curve

            fpr = fp / (fp + tn)
            fnr = fn / (fn + tp)
            tpr = tp / (fn + tp)

            fpr_matrix[i, j, k] = fpr
            fnr_matrix[i, j, k] = fnr
            tpr_matrix[i, j, k] = tpr

            new_row = {
            'Movement Threshold': param1,
            'Diff Threshold': param2,
            'ROI Threshold': param3,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'TPR': tpr,
            'FPR': fpr,
            'FNR': fnr,
            'Precision': precision,
            'Recall': recall,
            'AUC': roc_auc
            }
        
            rows_list.append(new_row)

new_data = pd.DataFrame(rows_list)
results = pd.concat([results, new_data], ignore_index=True)
results.to_excel(excel_file, index=False)      
        
# find the minmum intersection of FNR and FPR
sum_matrix = fpr_matrix + fnr_matrix
min_index = np.unravel_index(np.argmin(sum_matrix, axis=None), sum_matrix.shape)
best_movement_threshold = movement_threshold_values[min_index[0]]
best_diff_threshold = diff_threshold_values[min_index[1]]
best_roi_threshold = roi_threshold_values[min_index[2]]

print(f"The best movement threshold to minimize the FPR and FNR trade-off is: {best_movement_threshold}")
print(f"The best diff threshold to minimize the FPR and FNR trade-off is: {best_diff_threshold}")
print(f"The best roi threshold to minimize the FPR and FNR trade-off is: {best_roi_threshold}")

optimal_fpr = fpr_matrix[min_index[0], min_index[1], min_index[2]]
optimal_fnr = fnr_matrix[min_index[0], min_index[1], min_index[2]]
optimal_tpr = tpr_matrix[min_index[0], min_index[1], min_index[2]]

print(f"Optimal FPR (False Positive Rate) is: {optimal_fpr}")
print(f"Optimal FNR (False Negative Rate) is: {optimal_fnr}")

'''
threshold hierarchy

    highest hierarchy

        --movement-threshold: Describe the distance between bounding boxes of the same ID in two consecutive frames, beyond which exceeding it will lead to the detection of a novelty.

            (test range 30 - 100)

        --diff-threshold: Define the spatial distance between the bounding boxes of an appearing ID and a disappearing ID in two consecutive frames. This distance serves as the criterion for matching two IDs.

            (test range 10 - 50)

        --roi-threshold: Specify the minimum object area required for detection. Objects with an area below this threshold will not be considered as valid detections.

            (test range 500 - 1500)

        --buffer-size: The larger the better (No need to plot ROC curve and find the best one)
            
        --predict_size : The larger the better (No need to plot ROC curve and find the best one)

    lowest hierarchy

'''








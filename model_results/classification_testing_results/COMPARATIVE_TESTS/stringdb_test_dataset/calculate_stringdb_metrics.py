import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    f1_score, 
    roc_auc_score, 
    recall_score, 
    precision_score
)


df = pd.read_csv(r".\ppi_test_stringdb_check.csv")

# Actual and predicted labels
y_true = df['PPI_type']
y_pred = df['stringdb_check']
y_pred_prob = df['stringdb_score']

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Confusion matrix to derive specificity and sensitivity
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Specificity (True Negative Rate)
specificity = tn / (tn + fp)

# Sensitivity (Recall or True Positive Rate)
sensitivity = recall_score(y_true, y_pred)

# Precision
precision = precision_score(y_true, y_pred)

# F1 Score
f1 = f1_score(y_true, y_pred)

# F2 Score
f2 = (1 + 2**2) * (precision * sensitivity) / (2**2 * precision + sensitivity)

# ROCAUC Score
roc_auc = roc_auc_score(y_true, y_pred_prob)

# Printing the results
print(f'Accuracy: {accuracy:.8f}')
print(f'Specificity: {specificity:.8f}')
print(f'Sensitivity: {sensitivity:.8f}')
print(f'F1 Score: {f1:.8f}')
print(f'F2 Score: {f2:.8f}')
print(f'ROCAUC Score: {roc_auc:.8f}')

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

# Load the primary dataset
csv_file = r"C:\Users\harry\Desktop\TEST_1\FINAL_REDUCED_TEST_ALL_FEATURES_AND_PREDICTIONS.csv"
df = pd.read_csv(csv_file)

# Define columns where NaNs should be removed
columns_to_check = [
    "seq_A", "seq_B", "similarity_check", "label", 
    "First_Run_Best_Model", "First_Run_Ensemble_Model", 
    "Second_Run_Best_Model", "Second_Run_Ensemble_Model", 
    "ensemblA", "ensemblB"
]

# Remove NaNs in critical columns
df = df.dropna(subset=columns_to_check)

# Drop the 'PPI_type' column if it exists
if "PPI_type" in df.columns:
    df = df.drop(columns=["PPI_type"])

# Extract true labels
y_true = df["label"]

# Define model prediction columns
model_columns = [
    "First_Run_Best_Model",
    "First_Run_Ensemble_Model",
    "Second_Run_Best_Model",
    "Second_Run_Ensemble_Model"
]

# Remove rows where any y_pred column has NaN
df = df.dropna(subset=model_columns)

# Store model predictions
model_predictions = {col: df[col] for col in model_columns}

# Function to compute metrics
def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) != 0 else 0
    roc_auc = roc_auc_score(y_true, y_pred)

    # Compute Specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    return accuracy, specificity, recall, f1, f2, roc_auc

# Store metrics for both runs
metrics_results = {
    "Best_Model": [],
    "Ensemble_Model": []
}

# Compute metrics for each run
metrics_results["Best_Model"].append(compute_metrics(y_true[df.index], model_predictions["First_Run_Best_Model"]))
metrics_results["Best_Model"].append(compute_metrics(y_true[df.index], model_predictions["Second_Run_Best_Model"]))

metrics_results["Ensemble_Model"].append(compute_metrics(y_true[df.index], model_predictions["First_Run_Ensemble_Model"]))
metrics_results["Ensemble_Model"].append(compute_metrics(y_true[df.index], model_predictions["Second_Run_Ensemble_Model"]))

# Convert results to numpy arrays for easy computation of mean and std
metrics_results["Best_Model"] = np.array(metrics_results["Best_Model"])
metrics_results["Ensemble_Model"] = np.array(metrics_results["Ensemble_Model"])

# Load additional CSV and TSV files with no headers
csv_extra_file = r"C:\Users\harry\Desktop\TEST_1\test_notebooks\TEST_dscript_merged.predictions.csv"
tsv_extra_file = r"C:\Users\harry\Desktop\TEST_1\test_notebooks\TEST_TT_merged.predictions.tsv"

df_csv_extra = pd.read_csv(csv_extra_file, header=None)
df_tsv_extra = pd.read_csv(tsv_extra_file, header=None, sep="\t")

# Extract true labels and predicted probabilities
y_true_csv = df_csv_extra.iloc[:, 2]  # Column 3
y_pred_probs_csv = df_csv_extra.iloc[:, 3]  # Column 4
y_pred_csv = (y_pred_probs_csv >= 0.5).astype(int)  # Convert probabilities to binary predictions

y_true_tsv = df_tsv_extra.iloc[:, 2]  # Column 3
y_pred_probs_tsv = df_tsv_extra.iloc[:, 3]  # Column 4
y_pred_tsv = (y_pred_probs_tsv >= 0.5).astype(int)  # Convert probabilities to binary predictions

# Compute metrics for CSV and TSV datasets
csv_metrics = compute_metrics(y_true_csv, y_pred_csv)
tsv_metrics = compute_metrics(y_true_tsv, y_pred_tsv)

# Metrics names
metric_names = ["Accuracy", "Specificity", "Sensitivity (Recall)", "F1 Score", "F2 Score", "ROC-AUC Score"]

# Print results
print("\n=== EOA Test Results ===")
for model_type, metrics in metrics_results.items():
    print(f"\n{model_type} Metrics (Mean ± Std):")
    for i, metric_name in enumerate(metric_names):
        mean = np.mean(metrics[:, i])
        std = np.std(metrics[:, i])
        print(f"{metric_name}: {mean:.4f} ± {std:.4f}")

# Print results for additional datasets
print("\n=== DSCRIPT Test Results ===")
for i, metric_name in enumerate(metric_names):
    print(f"{metric_name}: {csv_metrics[i]:.4f}")

print("\n=== T&T Test Results ===")
for i, metric_name in enumerate(metric_names):
    print(f"{metric_name}: {tsv_metrics[i]:.4f}")

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'})

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
csv_path = os.path.join(project_root, 'data', 'tabular', 'test_data.csv')

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find dataset at: {csv_path}")

df = pd.read_csv(csv_path)

# --- 1. Class Distribution Histogram ---
plt.figure(figsize=(8, 5))
label_col = None
for col in ['Class_Label', 'label', 'target', 'Class']:
    if col in df.columns:
        label_col = col
        break

if label_col:
    sns.countplot(x=label_col, data=df, palette='viridis')
    plt.title('Class Distribution Across Stress Categories', fontsize=14, fontweight='bold')
    plt.xlabel('Stress Classes', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
else:
    sns.countplot(x=df.iloc[:, -1], palette='viridis')
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)

plt.tight_layout()
hist_path = os.path.join(project_root, 'class_distribution_histogram.png')
plt.savefig(hist_path, dpi=300)
plt.close()
print(f"Successfully generated histogram at: {hist_path}")

# --- 2. Correlation Heatmap (Fixed Color Scaling) ---
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['number'])

if numeric_df.empty:
    print("Error: No numeric columns found in test_data.csv!")
else:
    corr_matrix = numeric_df.corr()
    
    # vmin=-1 and vmax=1 ensures the color scale is fixed and mathematically correct
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5, cbar=True, vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap (Telemetry & Labels)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    heatmap_path = os.path.join(project_root, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Successfully generated fixed heatmap at: {heatmap_path}")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV files
df_local = pd.read_csv("model_metrics_log_Thursday.csv")
df_k8s = pd.read_csv("model_metrics_log_Thursday_k8s.csv")

# Adjust the timestamps to be in the same format
df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], format="%H:%M:%S").dt.time
df_k8s["timestamp"] = df_local["timestamp"]

# Create a 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Comparison of Model Performance (Local vs Kubernetes)", fontsize=14, fontweight="bold")

# Define the metrics, titles, and colors for the plots
metrics = ["accuracy", "f1", "recall", "precision"]
titles = ["Accuracy", "F1 Score", "Recall", "Precision"]
colors = ["#1f77b4", "#ff7f0e"]

# Select only some timestamps for better readability
selected_timestamps = df_local["timestamp"][::5].astype(str).tolist()

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    
    # Convert the data to the appropriate format
    x_values = df_local["timestamp"].astype(str).tolist()
    y_local = df_local[metric].astype(float).tolist()
    y_k8s = df_k8s[metric].astype(float).tolist()

    # Plot the data for both local and Kubernetes
    sns.lineplot(x=x_values, y=y_local, marker="o", markersize=4, linewidth=2, linestyle="-", label="Local", ax=ax, color=colors[0])
    sns.lineplot(x=x_values, y=y_k8s, marker="s", markersize=4, linewidth=2, linestyle="--", label="Kubernetes", ax=ax, color=colors[1])

    # Configure the plot
    ax.set_title(titles[i], fontsize=12, fontweight="bold")
    ax.set_xlabel("Timestamp", fontsize=10)
    ax.set_ylabel(metric.capitalize(), fontsize=10)

    # Use a dashed grid for better readability
    ax.grid(True, linestyle="dashed", alpha=0.6)

    # Set the x-axis labels to be the selected timestamps
    ax.set_xticks(selected_timestamps)
    ax.set_xticklabels(selected_timestamps, rotation=45, fontsize=9)

# Adjust the layout and show the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.05), fontsize=10)
plt.show()

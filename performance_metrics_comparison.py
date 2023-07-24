import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Performance metrics data for each model
performance_data = {
    'Model': ['SVM', 'BERT', 'XLNet', 'RoBERTa'],
    'Accuracy': [0.7670103092783506, 0.85, 0.90, 0.91],
    'Precision': [0.7686424278760001, 0.84, 0.88, 0.89],
    'Recall': [0.7670103092783506, 0.83, 0.91, 0.92],
    'F1-score': [0.7566496921593266, 0.83, 0.89, 0.91]
}

# Create a DataFrame from the input data
comparison_df = pd.DataFrame(performance_data)

# Set up the line plot
plt.figure(figsize=(10, 6))

# Plot lines for each performance metric
sns.lineplot(x='Model', y='Accuracy', data=comparison_df, marker='o', label='Accuracy', linewidth=2.5)
sns.lineplot(x='Model', y='Precision', data=comparison_df, marker='o', label='Precision', linewidth=2.5)
sns.lineplot(x='Model', y='Recall', data=comparison_df, marker='o', label='Recall', linewidth=2.5)
sns.lineplot(x='Model', y='F1-score', data=comparison_df, marker='o', label='F1-score', linewidth=2.5)

# Set plot limits and labels
plt.ylim(0.7, 1.0)  # Adjust the y-axis scale to focus on the range 0.7 to 1.0
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Performance Metrics Comparison')
plt.legend(title='Metrics', fontsize='medium', loc='lower right')  # Adjust the position of the legend to lower right

# Add value labels above each point
for i in range(len(comparison_df)):
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
        value = comparison_df.loc[i, metric]
        plt.text(i, value, f"{value:.2f}", ha='center', va='bottom', fontsize=10)

# Save the plot
plt.savefig('performance_metrics_comparison.png')
plt.show()

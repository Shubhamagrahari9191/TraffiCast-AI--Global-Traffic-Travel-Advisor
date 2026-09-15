import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_model_comparison(results_csv_path: str, dataset_name: str, output_dir: str = 'outputs/figures'):
    """
    Reads the results table and plots a bar chart comparison of models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(results_csv_path):
        print(f"Results CSV not found at {results_csv_path}. Skipping comparison plot.")
        return
        
    df = pd.read_csv(results_csv_path)
    
    # Check if we have the needed columns
    required_cols = {'Model', 'Horizon', 'MAE', 'RMSE', 'MAPE'}
    if not required_cols.issubset(df.columns):
        print(f"Results CSV is missing required columns. Found: {list(df.columns)}")
        return
        
    horizons = df['Horizon'].unique()
    models = df['Model'].unique()
    
    # Set up matplotlib style/parameters
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['MAE', 'RMSE', 'MAPE']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    x = np.arange(len(horizons))
    width = 0.12  # width of bar
    
    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx]
        
        # Plot bars for each model
        for i, model in enumerate(models):
            df_model = df[df['Model'] == model]
            
            # Make sure we sort by horizon (15, 30, 60 min)
            df_model = df_model.set_index('Horizon').reindex(horizons).reset_index()
            
            values = df_model[metric].values
            
            # Position offset
            offset = (i - len(models)/2) * width + width/2
            ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)], edgecolor='k', alpha=0.85)
            
        ax.set_title(f"{metric} Comparison ({dataset_name})", fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons, fontsize=11)
        ax.set_ylabel(metric if metric != 'MAPE' else 'MAPE (%)', fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if m_idx == 0:
            ax.legend(title="Models", loc='upper left')
            
    plt.suptitle(f"Model Performance Comparison Across Prediction Horizons ({dataset_name})", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    comparison_plot_path = os.path.join(output_dir, f"model_comparison_{dataset_name.lower()}.png")
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison chart to: {comparison_plot_path}")

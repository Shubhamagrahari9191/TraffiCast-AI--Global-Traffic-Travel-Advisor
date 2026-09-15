import os
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curves(train_losses, val_losses, model_name: str, dataset_name: str, output_dir: str = 'outputs/figures'):
    """
    Plots training and validation loss curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', color='darkorange', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='royalblue', linewidth=2)
    plt.title(f"{model_name.upper()} Loss Curves ({dataset_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{model_name.lower()}_{dataset_name.lower()}_loss_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to: {plot_path}")

def plot_actual_vs_predicted(y_true, y_pred, sensor_name: str, horizon_min: int, dataset_name: str, 
                              num_timesteps: int = 288, output_dir: str = 'outputs/figures'):
    """
    Plots a time-series comparison of actual vs predicted traffic speeds.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Slice the first 'num_timesteps' samples (288 = 1 day of data at 5-minute sampling)
    true_slice = y_true[:num_timesteps]
    pred_slice = y_pred[:num_timesteps]
    
    plt.figure(figsize=(12, 5))
    plt.plot(true_slice, label='Actual Speed', color='black', alpha=0.8, linewidth=1.5)
    plt.plot(pred_slice, label='Predicted Speed', color='crimson', linestyle='--', alpha=0.9, linewidth=1.5)
    plt.title(f"Actual vs Predicted Speed - Sensor {sensor_name} ({dataset_name})\nHorizon: {horizon_min} min ahead")
    plt.xlabel("Time Steps (5-min intervals)")
    plt.ylabel("Traffic Speed (mph)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"actual_vs_pred_{sensor_name}_{horizon_min}m_{dataset_name.lower()}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved actual vs predicted speed plot to: {plot_path}")

def plot_error_distribution(y_true, y_pred, model_name: str, dataset_name: str, output_dir: str = 'outputs/figures'):
    """
    Plots a histogram of prediction errors (residuals).
    """
    os.makedirs(output_dir, exist_ok=True)
    errors = (y_true - y_pred).flatten()
    
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color='teal', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
    plt.title(f"Prediction Error (Residuals) Distribution\n{model_name.upper()} ({dataset_name})")
    plt.xlabel("Prediction Error (Actual - Predicted)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{model_name.lower()}_{dataset_name.lower()}_error_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved error distribution to: {plot_path}")

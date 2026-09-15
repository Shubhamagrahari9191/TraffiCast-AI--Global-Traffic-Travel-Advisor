import os
import json
import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import yaml

def read_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class SHAPAnalyzer:
    """
    SHAP Analysis handler for the Hybrid Transformer-BiLSTM model.
    
    Methodology Note:
    Due to the computational complexity of Kernel SHAP over continuous spatiotemporal inputs
    (evaluating permutations across num_lags * num_sensors inputs), SHAP explanations are generated
    on a representative test subset using a bounded background dataset with a fixed random seed.
    This provides reproducible feature attribution without prohibitive runtime latency.
    """
    def __init__(self, model, background_data: np.ndarray, device: str = 'cpu', config: dict = None):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.config = config or read_config()
        
        shap_cfg = self.config.get('shap', {})
        self.num_bg = shap_cfg.get('num_background_samples', 100)
        self.num_test = shap_cfg.get('num_test_samples', 10)
        self.nsamples = shap_cfg.get('nsamples', 100)
        self.seed = shap_cfg.get('seed', 42)
        
        # Bounded background dataset drawn reproducibly
        rng = np.random.default_rng(seed=self.seed)
        bg_size = min(len(background_data), self.num_bg)
        bg_indices = rng.choice(len(background_data), size=bg_size, replace=False)
        self.background_data = background_data[bg_indices]

    def explain_prediction(self, test_data: np.ndarray, target_sensor: int = 0, target_horizon_idx: int = 0):
        """
        Computes SHAP values explaining predictions for a specific sensor and horizon on a bounded representative subset.
        
        Args:
            test_data (np.ndarray): Test sequences of shape (num_test_samples, input_window, num_sensors)
            target_sensor (int): Index of sensor to explain
            target_horizon_idx (int): Index of prediction horizon (0 = 15m, 1 = 30m, 2 = 60m)
            
        Returns:
            shap_values (np.ndarray): shape (num_test_samples, input_window, num_sensors)
            test_subset (np.ndarray): shape (num_test_samples, input_window, num_sensors)
        """
        num_test_samples, input_window, num_sensors = test_data.shape
        
        def model_wrapper(x_flat):
            x_3d = x_flat.reshape(-1, input_window, num_sensors)
            x_tensor = torch.tensor(x_3d, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.model(x_tensor)
                return out[:, target_horizon_idx, target_sensor].cpu().numpy()
                
        # Draw representative test subset reproducibly
        rng = np.random.default_rng(seed=self.seed)
        test_size = min(len(test_data), self.num_test)
        test_indices = rng.choice(len(test_data), size=test_size, replace=False)
        test_subset = test_data[test_indices]
        
        # Flatten background and test data for KernelExplainer
        background_flat = self.background_data.reshape(self.background_data.shape[0], -1)
        test_subset_flat = test_subset.reshape(test_subset.shape[0], -1)
        
        explainer = shap.KernelExplainer(model_wrapper, background_flat)
        
        print(f"Computing SHAP values (subset size={test_size}, background size={len(self.background_data)}, nsamples={self.nsamples})...")
        shap_values_flat = explainer.shap_values(test_subset_flat, nsamples=self.nsamples)
        
        if isinstance(shap_values_flat, list):
            shap_values_flat = shap_values_flat[0]
            
        shap_values_3d = shap_values_flat.reshape(test_subset.shape)
        
        return shap_values_3d, test_subset

    def plot_and_save_shap(self, shap_values, test_subset, target_sensor_name: str, 
                           horizon_min: int, output_dir: str = 'outputs/figures'):
        """
        Generates and saves publication-quality SHAP plots with honest representative metadata.
        """
        os.makedirs(output_dir, exist_ok=True)
        num_samples, input_window, num_sensors = test_subset.shape
        feature_names = [f"t-{input_window - i}" for i in range(input_window)]
        
        # 1. Global Temporal Feature Importance Plot
        avg_shap_temporal = np.mean(np.abs(shap_values), axis=2)
        mean_abs_shap = np.mean(avg_shap_temporal, axis=0)
        
        plt.figure(figsize=(8, 5))
        plt.bar(feature_names, mean_abs_shap, color='royalblue', edgecolor='k', alpha=0.8)
        plt.title(f"Representative Temporal Feature Importance\n(Sensor: {target_sensor_name}, Horizon: {horizon_min}m)")
        plt.xlabel("Historical Timesteps")
        plt.ylabel("Mean |SHAP Value| (Impact on Model Output)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        temporal_plot_path = os.path.join(output_dir, f"shap_temporal_importance_{target_sensor_name}_{horizon_min}m.png")
        plt.savefig(temporal_plot_path, dpi=300)
        plt.close()
        print(f"Saved temporal SHAP importance to: {temporal_plot_path}")
        
        # 2. Local Explanation Bar Plot for Sample 0
        plt.figure(figsize=(8, 5))
        local_shap = shap_values[0, :, 0]
        colors = ['crimson' if val >= 0 else 'dodgerblue' for val in local_shap]
        
        plt.bar(feature_names, local_shap, color=colors, edgecolor='k', alpha=0.8)
        plt.title(f"Representative Local SHAP Explanation (Sample 0)\n(Sensor: {target_sensor_name}, Horizon: {horizon_min}m)")
        plt.xlabel("Historical Timesteps")
        plt.ylabel("SHAP Value (Impact on Prediction)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        local_plot_path = os.path.join(output_dir, f"shap_local_explanation_{target_sensor_name}_{horizon_min}m.png")
        plt.savefig(local_plot_path, dpi=300)
        plt.close()
        print(f"Saved local SHAP explanation to: {local_plot_path}")
        
        # 3. SHAP Summary Plot
        flat_shap = shap_values.reshape(num_samples, -1)
        flat_data = test_subset.reshape(num_samples, -1)
        
        flat_feature_names = []
        for i in range(input_window):
            for j in range(num_sensors):
                flat_feature_names.append(f"t-{input_window - i} (S{j})")
                
        plt.figure(figsize=(10, 6))
        shap.summary_plot(flat_shap, flat_data, feature_names=flat_feature_names, max_display=15, show=False)
        plt.title(f"Representative SHAP Summary Plot (Top 15 Features)\nHorizon: {horizon_min}m", fontsize=12)
        plt.tight_layout()
        
        summary_plot_path = os.path.join(output_dir, f"shap_summary_plot_{horizon_min}m.png")
        plt.savefig(summary_plot_path, dpi=300)
        plt.close()
        print(f"Saved SHAP summary plot to: {summary_plot_path}")
        
        # Save metadata record
        meta_path = os.path.join(output_dir, f"shap_metadata_{target_sensor_name}_{horizon_min}m.json")
        with open(meta_path, 'w') as f:
            json.dump({
                'representative_sensor': str(target_sensor_name),
                'horizon_minutes': horizon_min,
                'background_samples': len(self.background_data),
                'evaluated_test_subset_size': num_samples,
                'nsamples_perturbations': self.nsamples,
                'methodology': "Representative test subset with bounded background dataset (Kernel SHAP)"
            }, f, indent=2)

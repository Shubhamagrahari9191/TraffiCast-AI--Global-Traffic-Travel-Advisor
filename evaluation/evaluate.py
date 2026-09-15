import os
import sys
import numpy as np
import pandas as pd
import yaml

# Add root folder to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import compute_all_metrics
from evaluation.statistical_tests import perform_wilcoxon_test

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_evaluation(dataset_name=None):
    config = load_config()
    if dataset_name:
        config['dataset']['name'] = dataset_name
        
    dataset = config['dataset']['name']
    
    print("=" * 65)
    print(f"Statistical Significance Testing per Horizon ({dataset})")
    print("=" * 65)
    
    # 1. Load Proposed model predictions and ground truths
    hybrid_preds_path = f"outputs/predictions/hybrid_{dataset}_preds.npy"
    hybrid_targets_path = f"outputs/predictions/hybrid_{dataset}_targets.npy"
    
    if not os.path.exists(hybrid_preds_path) or not os.path.exists(hybrid_targets_path):
        print("Proposed hybrid model predictions not found. Please train and evaluate all models first:")
        print("python main.py --experiment baselines")
        return
        
    y_pred_proposed = np.load(hybrid_preds_path)
    y_true = np.load(hybrid_targets_path)
    
    # Standard horizons evaluated in project
    horizons = ['15m', '30m', '60m']
    baselines = ['lstm', 'gru', 'bilstm', 'transformer', 'arima']
    significance_results = []
    
    for baseline in baselines:
        baseline_preds_path = f"outputs/predictions/{baseline}_{dataset}_preds.npy"
        
        if baseline == 'arima':
            baseline_targets_path = f"outputs/predictions/arima_{dataset}_targets.npy"
            if not os.path.exists(baseline_preds_path) or not os.path.exists(baseline_targets_path):
                print(f"ARIMA predictions not found for {dataset}. Skipping ARIMA significance testing.")
                continue
                
            y_pred_base = np.load(baseline_preds_path)
            y_true_base = np.load(baseline_targets_path)
            
            arima_config = config.get('arima', {})
            step_size = arima_config.get('step_size', 50 if not arima_config.get('full_experiment', False) else 200)
            
            # Align proposed model predictions to the exact timestamps and sensors evaluated by ARIMA
            num_arima_samples = y_pred_base.shape[0]
            sampled_indices = list(range(0, y_pred_proposed.shape[0], step_size))[:num_arima_samples]
            
            sensors = arima_config.get('selected_sensors', [0, 1, 2, 3, 4])
            if arima_config.get('full_experiment', False):
                sensors = list(range(y_true.shape[2]))
                
            y_pred_proposed_aligned = y_pred_proposed[sampled_indices][:, :, sensors]
            y_true_aligned = y_true[sampled_indices][:, :, sensors]
            
            print(f"\nEvaluating Baseline: ARIMA (Sampled grid, {len(sensors)} sensors, step={step_size})")
            for h_idx, h_name in enumerate(horizons):
                t_slice = y_true_aligned[:, h_idx, :]
                b_slice = y_pred_base[:, h_idx, :]
                p_slice = y_pred_proposed_aligned[:, h_idx, :]
                
                test_res = perform_wilcoxon_test(t_slice, b_slice, p_slice, max_samples=10000, seed=42)
                
                print(f"  [{h_name}] Proposed Hybrid vs ARIMA:")
                print(f"    - Wilcoxon Statistic: {test_res['statistic']:.2f} | p-value: {test_res['p_value']:.2e}")
                print(f"    - Significant (alpha=0.05): {test_res['is_significant']}")
                print(f"    - Evaluated pairs: {test_res['evaluated_sample_size']} (Subsampled: {test_res['subsampled']})")
                
                significance_results.append({
                    'Baseline_Model': 'ARIMA',
                    'Horizon': h_name,
                    'Wilcoxon_Statistic': test_res['statistic'],
                    'p_value': test_res['p_value'],
                    'Significant_0_05': test_res['is_significant'],
                    'Evaluated_Pairs': test_res['evaluated_sample_size'],
                    'Interpretation': test_res['interpretation']
                })
        else:
            if not os.path.exists(baseline_preds_path):
                print(f"Predictions for baseline '{baseline}' not found. Skipping.")
                continue
                
            y_pred_base = np.load(baseline_preds_path)
            print(f"\nEvaluating Baseline: {baseline.upper()}")
            
            for h_idx, h_name in enumerate(horizons):
                t_slice = y_true[:, h_idx, :]
                b_slice = y_pred_base[:, h_idx, :]
                p_slice = y_pred_proposed[:, h_idx, :]
                
                test_res = perform_wilcoxon_test(t_slice, b_slice, p_slice, max_samples=10000, seed=42)
                
                print(f"  [{h_name}] Proposed Hybrid vs {baseline.upper()}:")
                print(f"    - Wilcoxon Statistic: {test_res['statistic']:.2f} | p-value: {test_res['p_value']:.2e}")
                print(f"    - Significant (alpha=0.05): {test_res['is_significant']}")
                print(f"    - Evaluated pairs: {test_res['evaluated_sample_size']} (Subsampled: {test_res['subsampled']})")
                
                significance_results.append({
                    'Baseline_Model': baseline.upper(),
                    'Horizon': h_name,
                    'Wilcoxon_Statistic': test_res['statistic'],
                    'p_value': test_res['p_value'],
                    'Significant_0_05': test_res['is_significant'],
                    'Evaluated_Pairs': test_res['evaluated_sample_size'],
                    'Interpretation': test_res['interpretation']
                })
                
    if len(significance_results) > 0:
        df_sig = pd.DataFrame(significance_results)
        csv_path = f"outputs/tables/statistical_significance_{dataset}.csv"
        df_sig.to_csv(csv_path, index=False)
        print(f"\nStatistical significance table saved to: {csv_path}")
        print("=" * 65)

if __name__ == '__main__':
    run_evaluation()

import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd
import torch
from scipy import stats

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import compute_all_metrics
from evaluation.statistical_tests import perform_wilcoxon_test
from models.transformer_bilstm import TransformerBiLSTMModel

def print_header(title):
    print("\n" + "=" * 70)
    print(f"AUDIT STEP: {title}")
    print("=" * 70)

def audit_npy_predictions():
    print_header("1. RAW NPY PREDICTION ARRAYS INTEGRITY AUDIT")
    datasets = ['METR-LA', 'PEMS04']
    models = ['hybrid', 'lstm', 'gru', 'bilstm', 'transformer', 'arima']
    
    audit_report = {}
    
    for ds in datasets:
        audit_report[ds] = {}
        print(f"\n--- Dataset: {ds} ---")
        for m in models:
            preds_file = f"outputs/predictions/{m}_{ds}_preds.npy"
            targets_file = f"outputs/predictions/{m}_{ds}_targets.npy"
            
            if not os.path.exists(preds_file):
                print(f"❌ Missing predictions: {preds_file}")
                continue
                
            preds = np.load(preds_file)
            audit_report[ds][m] = {
                'preds_shape': preds.shape,
                'preds_nan_count': int(np.isnan(preds).sum()),
                'preds_min': float(np.min(preds)),
                'preds_max': float(np.max(preds)),
                'preds_mean': float(np.mean(preds)),
                'preds_std': float(np.std(preds)),
            }
            
            has_targets = os.path.exists(targets_file)
            if has_targets:
                targets = np.load(targets_file)
                audit_report[ds][m]['targets_shape'] = targets.shape
                audit_report[ds][m]['targets_nan_count'] = int(np.isnan(targets).sum())
                audit_report[ds][m]['targets_min'] = float(np.min(targets))
                audit_report[ds][m]['targets_max'] = float(np.max(targets))
                audit_report[ds][m]['targets_mean'] = float(np.mean(targets))
            
            print(f"[{m.upper()}] Shape: {preds.shape} | NaNs: {audit_report[ds][m]['preds_nan_count']} | Range: [{audit_report[ds][m]['preds_min']:.2f}, {audit_report[ds][m]['preds_max']:.2f}] | Mean: {audit_report[ds][m]['preds_mean']:.2f}")
            
    return audit_report

def audit_metrics_csv_vs_raw_arrays():
    print_header("2. METRICS VERIFICATION (CSV vs RECOMPUTED FROM RAW NPY)")
    datasets = ['METR-LA', 'PEMS04']
    models = ['HYBRID', 'LSTM', 'GRU', 'BILSTM', 'TRANSFORMER', 'ARIMA']
    horizons = [15, 30, 60]
    
    for ds in datasets:
        csv_path = f"outputs/tables/baseline_results_{ds}.csv"
        print(f"\n--- Checking CSV: {csv_path} ---")
        if not os.path.exists(csv_path):
            print(f"❌ File missing: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        
        # Load targets
        hybrid_targets = np.load(f"outputs/predictions/hybrid_{ds}_targets.npy")
        arima_targets = np.load(f"outputs/predictions/arima_{ds}_targets.npy")
        
        for m in models:
            preds_file = f"outputs/predictions/{m.lower()}_{ds}_preds.npy"
            if not os.path.exists(preds_file):
                continue
            preds = np.load(preds_file)
            targets = arima_targets if m == 'ARIMA' else hybrid_targets
            
            for h_idx, h_min in enumerate(horizons):
                # Slices
                t_slice = targets[:, h_idx, :]
                p_slice = preds[:, h_idx, :]
                recomputed = compute_all_metrics(t_slice, p_slice)
                
                # Match either exact name or TRANSFORMER+BiLSTM
                if m == 'HYBRID':
                    row = df[(df['Model'].isin(['HYBRID', 'TRANSFORMER+BiLSTM'])) & (df['Horizon'].str.contains(str(h_min)))]
                else:
                    row = df[(df['Model'] == m) & (df['Horizon'].str.contains(str(h_min)))]
                    
                if len(row) == 0:
                    print(f"❌ Missing CSV row for {m} at {h_min}m")
                    continue
                    
                csv_mae = float(row['MAE'].values[0])
                csv_rmse = float(row['RMSE'].values[0])
                csv_mape = float(row['MAPE'].values[0])
                
                mae_diff = abs(recomputed['MAE'] - csv_mae)
                rmse_diff = abs(recomputed['RMSE'] - csv_rmse)
                mape_diff = abs(recomputed['MAPE'] - csv_mape)
                
                status = "✅ PASS" if (mae_diff < 1e-3 and rmse_diff < 1e-3 and mape_diff < 1e-3) else "❌ MISMATCH"
                print(f"  {m} ({h_min}m): CSV MAE={csv_mae:.4f} vs Recalc={recomputed['MAE']:.4f} (Diff: {mae_diff:.2e}) -> {status}")

def audit_wilcoxon_csv_vs_recomputed():
    print_header("3. WILCOXON TEST VERIFICATION (CSV vs RECOMPUTED WITH SEED 42)")
    datasets = ['METR-LA', 'PEMS04']
    baselines = ['LSTM', 'GRU', 'BILSTM', 'TRANSFORMER', 'ARIMA']
    horizons = ['15m', '30m', '60m']
    
    for ds in datasets:
        csv_path = f"outputs/tables/statistical_significance_{ds}.csv"
        print(f"\n--- Checking Significance Table: {csv_path} ---")
        if not os.path.exists(csv_path):
            print(f"❌ File missing: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        
        hybrid_preds = np.load(f"outputs/predictions/hybrid_{ds}_preds.npy")
        hybrid_targets = np.load(f"outputs/predictions/hybrid_{ds}_targets.npy")
        
        for b in baselines:
            preds_file = f"outputs/predictions/{b.lower()}_{ds}_preds.npy"
            if not os.path.exists(preds_file):
                continue
            base_preds = np.load(preds_file)
            
            if b == 'ARIMA':
                base_targets = np.load(f"outputs/predictions/arima_{ds}_targets.npy")
                step_size = 50
                num_arima = base_preds.shape[0]
                sampled_indices = list(range(0, hybrid_preds.shape[0], step_size))[:num_arima]
                sensors = [0, 1, 2, 3, 4]
                
                h_preds_aligned = hybrid_preds[sampled_indices][:, :, sensors]
                h_targets_aligned = hybrid_targets[sampled_indices][:, :, sensors]
                
                for h_idx, h_name in enumerate(horizons):
                    test_res = perform_wilcoxon_test(
                        h_targets_aligned[:, h_idx, :], 
                        base_preds[:, h_idx, :], 
                        h_preds_aligned[:, h_idx, :],
                        max_samples=10000,
                        seed=42
                    )
                    row = df[(df['Baseline_Model'] == b) & (df['Horizon'] == h_name)]
                    if len(row) == 0:
                        print(f"❌ Missing row for {b} {h_name}")
                        continue
                    csv_stat = float(row['Wilcoxon_Statistic'].values[0])
                    csv_p = float(row['p_value'].values[0])
                    stat_diff = abs(test_res['statistic'] - csv_stat)
                    p_diff = abs(test_res['p_value'] - csv_p)
                    
                    status = "✅ PASS" if (stat_diff < 1e-2 and p_diff < 1e-4) else "❌ MISMATCH"
                    print(f"  ARIMA ({h_name}): CSV stat={csv_stat:.0f}, p={csv_p:.2e} | Recalc stat={test_res['statistic']:.0f}, p={test_res['p_value']:.2e} -> {status}")
            else:
                for h_idx, h_name in enumerate(horizons):
                    test_res = perform_wilcoxon_test(
                        hybrid_targets[:, h_idx, :], 
                        base_preds[:, h_idx, :], 
                        hybrid_preds[:, h_idx, :],
                        max_samples=10000,
                        seed=42
                    )
                    row = df[(df['Baseline_Model'] == b) & (df['Horizon'] == h_name)]
                    if len(row) == 0:
                        print(f"❌ Missing row for {b} {h_name}")
                        continue
                    csv_stat = float(row['Wilcoxon_Statistic'].values[0])
                    csv_p = float(row['p_value'].values[0])
                    stat_diff = abs(test_res['statistic'] - csv_stat)
                    p_diff = abs(test_res['p_value'] - csv_p)
                    
                    status = "✅ PASS" if (stat_diff < 1e-2 and p_diff < 1e-4) else "❌ MISMATCH"
                    print(f"  {b} ({h_name}): CSV stat={csv_stat:.0f}, p={csv_p:.2e} | Recalc stat={test_res['statistic']:.0f}, p={test_res['p_value']:.2e} -> {status}")

def audit_arima_audit_json():
    print_header("4. ARIMA FALLBACK AUDIT JSON VERIFICATION")
    datasets = ['METR-LA', 'PEMS04']
    for ds in datasets:
        path = f"outputs/tables/arima_fallback_statistics_{ds}.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"[{ds}] File exists: {path}")
            print(f"  - Order: {data.get('order')}")
            print(f"  - Total Predictions: {data.get('total_horizon_predictions')}")
            print(f"  - Fallbacks: {data.get('fallback_predictions')}")
            print(f"  - Fallback Rate: {data.get('fallback_percentage')}%")
            print(f"  - Sensors Evaluated: {data.get('total_sensors_evaluated')}")
            print(f"  - Step Size: {data.get('step_size')}")
        else:
            print(f"❌ Missing audit file: {path}")

def audit_model_architecture():
    print_header("5. MODEL FORWARD PASS & TENSOR DIMENSION AUDIT")
    model = TransformerBiLSTMModel(
        num_sensors=207,
        num_horizons=3,
        d_model=64,
        num_heads=4,
        transformer_layers=2,
        dim_feedforward=128,
        bilstm_hidden=64,
        bilstm_layers=1,
        dropout=0.2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Test batch forward pass
    dummy_input = torch.randn(16, 12, 207)
    out = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    assert out.shape == (16, 3, 207), f"Expected shape (16, 3, 207) but got {out.shape}"
    print("✅ Forward pass matches theoretical tensor dimensions exactly (B, num_horizons, num_sensors)!")

def audit_shap_figures():
    print_header("6. SHAP EXPLAINABILITY FIGURES AUDIT")
    figures_dir = "outputs/figures"
    expected_files = [
        "shap_summary_plot_15m.png",
        "shap_temporal_importance_Sensor 773869_15m.png",
        "shap_local_explanation_Sensor 773869_15m.png",
        "shap_temporal_importance_Sensor 0_15m.png",
        "shap_local_explanation_Sensor 0_15m.png"
    ]
    for ef in expected_files:
        fp = os.path.join(figures_dir, ef)
        if os.path.exists(fp):
            size_kb = os.path.getsize(fp) / 1024
            print(f"✅ Found: {ef} ({size_kb:.1f} KB)")
        else:
            print(f"❌ Missing: {ef}")

def audit_flask_api_live():
    print_header("7. LIVE FLASK REST API ENDPOINTS AUDIT")
    endpoints = [
        "http://127.0.0.1:5000/api/config?dataset=METR-LA",
        "http://127.0.0.1:5000/api/config?dataset=PEMS04",
        "http://127.0.0.1:5000/api/metrics?dataset=METR-LA",
        "http://127.0.0.1:5000/api/metrics?dataset=PEMS04",
        "http://127.0.0.1:5000/api/ablation?dataset=METR-LA",
        "http://127.0.0.1:5000/api/ablation?dataset=PEMS04",
        "http://127.0.0.1:5000/api/predictions?dataset=METR-LA&sensor=0&horizon=0",
        "http://127.0.0.1:5000/api/predictions?dataset=PEMS04&sensor=0&horizon=0"
    ]
    
    for url in endpoints:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                status_code = response.getcode()
                raw_data = response.read()
                data = json.loads(raw_data)
                
                # Check response payload summary
                keys = list(data.keys()) if isinstance(data, dict) else f"list({len(data)})"
                print(f"✅ {url} -> HTTP {status_code} | Payload keys/size: {keys}")
        except Exception as e:
            print(f"❌ {url} -> FAILED: {e}")

if __name__ == '__main__':
    audit_npy_predictions()
    audit_metrics_csv_vs_raw_arrays()
    audit_wilcoxon_csv_vs_recomputed()
    audit_arima_audit_json()
    audit_model_architecture()
    audit_shap_figures()
    audit_flask_api_live()

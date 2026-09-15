import os
import sys
import yaml
import random
import torch
import numpy as np
import pandas as pd

# Add root folder to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.create_sequences import get_train_val_test_data
from models.lstm import LSTMModel
from models.gru import GRUModel
from models.bilstm import BiLSTMModel
from models.transformer import TransformerModel
from models.arima import ARIMABaseline
from training.trainer import Trainer
from evaluation.metrics import compute_all_metrics

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # For MPS/CUDA speed

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_baselines(dataset_name=None, model_to_run=None):
    config = load_config()
    
    # Override dataset if provided via CLI
    if dataset_name:
        config['dataset']['name'] = dataset_name
        
    dataset = config['dataset']['name']
    raw_dir = config['dataset']['raw_dir']
    
    # 1. Seeding
    set_seeds(config['experiment']['seed'])
    
    # 2. Get dataset file path
    from preprocessing.load_data import resolve_dataset_path
    file_path = resolve_dataset_path(raw_dir, dataset)
    
    print(f"Loading and preprocessing data for {dataset}...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler = get_train_val_test_data(
        file_path=file_path,
        input_window=config['sequence']['input_window'],
        horizons=[3, 6, 12],  # correspond to 15m, 30m, 60m
        train_ratio=config['experiment']['train_ratio'],
        val_ratio=config['experiment']['val_ratio'],
        test_ratio=config['experiment']['test_ratio']
    )
    
    num_sensors = X_train.shape[2]
    print(f"Dataset Loaded successfully:")
    print(f"X_train shape: {X_train.shape} | Y_train shape: {Y_train.shape}")
    print(f"X_test shape:  {X_test.shape} | Y_test shape:  {Y_test.shape}")
    
    # Define baselines list
    baselines = {
        'lstm': lambda: LSTMModel(
            num_sensors=num_sensors,
            num_horizons=3,
            hidden_size=config['model']['lstm_hidden'],
            num_layers=config['model']['lstm_layers'],
            dropout=config['model']['dropout']
        ),
        'gru': lambda: GRUModel(
            num_sensors=num_sensors,
            num_horizons=3,
            hidden_size=config['model']['gru_hidden'],
            num_layers=config['model']['gru_layers'],
            dropout=config['model']['dropout']
        ),
        'bilstm': lambda: BiLSTMModel(
            num_sensors=num_sensors,
            num_horizons=3,
            hidden_size=config['model']['bilstm_hidden'],
            num_layers=config['model']['bilstm_layers'],
            dropout=config['model']['dropout']
        ),
        'transformer': lambda: TransformerModel(
            num_sensors=num_sensors,
            num_horizons=3,
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            transformer_layers=config['model']['transformer_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        )
    }
    
    # Determine models to evaluate
    if model_to_run:
        if model_to_run in baselines:
            models_eval = {model_to_run: baselines[model_to_run]}
        elif model_to_run == 'arima':
            models_eval = {'arima': None}
        else:
            print(f"Unknown model name: {model_to_run}")
            return
    else:
        # Run all
        models_eval = baselines.copy()
        models_eval['arima'] = None
        
    results_list = []
    
    # Ensure directories exist
    os.makedirs('outputs/tables', exist_ok=True)
    os.makedirs('outputs/predictions', exist_ok=True)
    
    for model_name, model_fn in models_eval.items():
        print("=" * 60)
        print(f"Evaluating Model: {model_name.upper()}")
        print("=" * 60)
        
        if model_name == 'arima':
            # ARIMA is independent of PyTorch, fits directly on train sequences
            arima_config = config['arima']
            # Select sensors to run ARIMA (running on all is too slow)
            sensors = arima_config['selected_sensors']
            if arima_config['full_experiment']:
                sensors = list(range(num_sensors))
                
            arima_baseline = ARIMABaseline()
            
            # Inverse-transform raw train/test split to run ARIMA on scale-accurate speeds
            # Get original values for accurate ARIMA fitting
            # Standard sequence split splits normalized arrays
            # We reconstruct train_raw and test_raw using scaler inverse_transform
            train_raw = scaler.inverse_transform(X_train[:, 0, :])  # (train_len, num_sensors)
            test_raw = scaler.inverse_transform(X_test[:, 0, :])    # (test_len, num_sensors)
            
            preds_norm_list = []
            targets_norm_list = []
            
            # Run ARIMA
            # step_size determines temporal sampling for prediction speed
            step_size = arima_config.get('step_size', 50 if not arima_config['full_experiment'] else 200)
            preds_raw, targets_raw = arima_baseline.predict(
                train_data=train_raw,
                test_data=test_raw,
                input_window=config['sequence']['input_window'],
                horizons=[3, 6, 12],
                sensor_indices=sensors,
                step_size=step_size,
                dataset_name=dataset
            )
            
            # Evaluate metrics sensor-wise and horizon-wise
            # horizons correspond to: 15m, 30m, 60m
            for h_idx, horizon_min in enumerate([15, 30, 60]):
                p_h = preds_raw[:, h_idx, :]
                t_h = targets_raw[:, h_idx, :]
                
                metrics = compute_all_metrics(t_h, p_h)
                print(f"ARIMA | Horizon {horizon_min}m | MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | MAPE: {metrics['MAPE']:.4f}%")
                
                results_list.append({
                    'Model': 'ARIMA',
                    'Horizon': f'{horizon_min} min',
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'sMAPE': metrics['sMAPE']
                })
                
            # Save predictions
            np.save(f"outputs/predictions/arima_{dataset}_preds.npy", preds_raw)
            np.save(f"outputs/predictions/arima_{dataset}_targets.npy", targets_raw)
            
        else:
            # DL Models
            model = model_fn()
            
            # Training Wrapper
            trainer = Trainer(
                model=model,
                device=config['experiment']['device'],
                learning_rate=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
                grad_clip=config['training']['grad_clip'],
                patience=config['training']['patience']
            )
            
            # Fit
            train_losses, val_losses = trainer.fit(
                X_train=X_train,
                Y_train=Y_train,
                X_val=X_val,
                Y_val=Y_val,
                epochs=config['training']['epochs'],
                batch_size=config['training']['batch_size'],
                model_name=f"{model_name}_{dataset}.pt"
            )
            
            # Predict
            preds_norm = trainer.predict(X_test, batch_size=config['training']['batch_size'])
            
            # Denormalize predictions and ground truths for fair evaluation
            # preds_norm shape: (num_samples, 3, num_sensors)
            # Y_test shape: (num_samples, 3, num_sensors)
            preds_raw = np.zeros_like(preds_norm)
            targets_raw = np.zeros_like(Y_test)
            
            for h in range(3):
                preds_raw[:, h, :] = scaler.inverse_transform(preds_norm[:, h, :])
                targets_raw[:, h, :] = scaler.inverse_transform(Y_test[:, h, :])
                
            # Save predictions
            np.save(f"outputs/predictions/{model_name}_{dataset}_preds.npy", preds_raw)
            np.save(f"outputs/predictions/{model_name}_{dataset}_targets.npy", targets_raw)
            
            # Compute Metrics per Horizon
            for h_idx, horizon_min in enumerate([15, 30, 60]):
                p_h = preds_raw[:, h_idx, :]
                t_h = targets_raw[:, h_idx, :]
                
                metrics = compute_all_metrics(t_h, p_h)
                print(f"{model_name.upper()} | Horizon {horizon_min}m | MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | MAPE: {metrics['MAPE']:.4f}%")
                
                results_list.append({
                    'Model': model_name.upper(),
                    'Horizon': f'{horizon_min} min',
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'sMAPE': metrics['sMAPE']
                })
                
    # Save results to tables
    df_results = pd.DataFrame(results_list)
    csv_path = f"outputs/tables/baseline_results_{dataset}.csv"
    
    # If the file already exists, we might merge/update it
    if os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            # Combine and remove duplicates
            df_results = pd.concat([df_old, df_results]).drop_duplicates(subset=['Model', 'Horizon'], keep='last')
        except Exception:
            pass
            
    df_results.to_csv(csv_path, index=False)
    print(f"Results table saved/updated at: {csv_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline models evaluation")
    parser.add_argument('--dataset', type=str, default=None, help="Dataset name")
    parser.add_argument('--model', type=str, default=None, help="Specific baseline model to evaluate")
    args = parser.parse_args()
    
    run_baselines(dataset_name=args.dataset, model_to_run=args.model)

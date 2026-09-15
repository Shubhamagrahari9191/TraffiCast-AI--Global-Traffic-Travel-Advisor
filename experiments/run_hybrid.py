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
from models.transformer_bilstm import TransformerBiLSTMModel
from training.trainer import Trainer
from evaluation.metrics import compute_all_metrics

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_hybrid(dataset_name=None):
    config = load_config()
    
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
        horizons=[3, 6, 12],
        train_ratio=config['experiment']['train_ratio'],
        val_ratio=config['experiment']['val_ratio'],
        test_ratio=config['experiment']['test_ratio']
    )
    
    num_sensors = X_train.shape[2]
    
    # Instantiate hybrid model
    model = TransformerBiLSTMModel(
        num_sensors=num_sensors,
        num_horizons=3,
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        transformer_layers=config['model']['transformer_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        bilstm_hidden=config['model']['bilstm_hidden'],
        bilstm_layers=config['model']['bilstm_layers'],
        dropout=config['model']['dropout']
    )
    
    print("=" * 60)
    print("Evaluating Proposed Model: TRANSFORMER-BiLSTM")
    print("=" * 60)
    
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
        model_name=f"hybrid_{dataset}.pt"
    )
    
    # Predict
    preds_norm = trainer.predict(X_test, batch_size=config['training']['batch_size'])
    
    # Denormalize
    preds_raw = np.zeros_like(preds_norm)
    targets_raw = np.zeros_like(Y_test)
    
    for h in range(3):
        preds_raw[:, h, :] = scaler.inverse_transform(preds_norm[:, h, :])
        targets_raw[:, h, :] = scaler.inverse_transform(Y_test[:, h, :])
        
    # Save predictions
    os.makedirs('outputs/predictions', exist_ok=True)
    np.save(f"outputs/predictions/hybrid_{dataset}_preds.npy", preds_raw)
    np.save(f"outputs/predictions/hybrid_{dataset}_targets.npy", targets_raw)
    
    results_list = []
    
    # Compute Metrics per Horizon
    for h_idx, horizon_min in enumerate([15, 30, 60]):
        p_h = preds_raw[:, h_idx, :]
        t_h = targets_raw[:, h_idx, :]
        
        metrics = compute_all_metrics(t_h, p_h)
        print(f"HYBRID | Horizon {horizon_min}m | MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | MAPE: {metrics['MAPE']:.4f}%")
        
        results_list.append({
            'Model': 'TRANSFORMER+BiLSTM',
            'Horizon': f'{horizon_min} min',
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'sMAPE': metrics['sMAPE']
        })
        
    # Save results to tables
    df_results = pd.DataFrame(results_list)
    csv_path = f"outputs/tables/baseline_results_{dataset}.csv"
    
    os.makedirs('outputs/tables', exist_ok=True)
    if os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            # Combine and remove duplicates
            df_results = pd.concat([df_old, df_results]).drop_duplicates(subset=['Model', 'Horizon'], keep='last')
        except Exception:
            pass
            
    df_results.to_csv(csv_path, index=False)
    print(f"Results table updated at: {csv_path}")

if __name__ == '__main__':
    run_hybrid()

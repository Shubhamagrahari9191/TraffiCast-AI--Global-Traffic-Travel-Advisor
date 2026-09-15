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
from models.bilstm import BiLSTMModel
from models.transformer import TransformerModel
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

def run_ablation_study(dataset_name=None):
    config = load_config()
    
    if dataset_name:
        config['dataset']['name'] = dataset_name
        
    dataset = config['dataset']['name']
    raw_dir = config['dataset']['raw_dir']
    
    # 1. Seeding
    set_seeds(config['experiment']['seed'])
    
    # 2. Load dataset
    from preprocessing.load_data import resolve_dataset_path
    file_path = resolve_dataset_path(raw_dir, dataset)
    
    print(f"Loading and preprocessing data for {dataset} (Ablation study)...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler = get_train_val_test_data(
        file_path=file_path,
        input_window=config['sequence']['input_window'],
        horizons=[3, 6, 12],
        train_ratio=config['experiment']['train_ratio'],
        val_ratio=config['experiment']['val_ratio'],
        test_ratio=config['experiment']['test_ratio']
    )
    
    num_sensors = X_train.shape[2]
    
    # Define Ablation experiments
    # To make this runnable in reasonable time, we use a slightly reduced epoch limit for hyperparameter variations
    ablation_epochs = min(15, config['training']['epochs'])  # 15 epochs for hyperparameter search is standard/sufficient
    
    experiments = {
        # Core Architecture Comparisons
        'BiLSTM Only': lambda: BiLSTMModel(
            num_sensors=num_sensors,
            num_horizons=3,
            hidden_size=config['model']['bilstm_hidden'],
            num_layers=config['model']['bilstm_layers'],
            dropout=config['model']['dropout']
        ),
        'Transformer Only': lambda: TransformerModel(
            num_sensors=num_sensors,
            num_horizons=3,
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            transformer_layers=config['model']['transformer_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout']
        ),
        'Proposed Hybrid': lambda: TransformerBiLSTMModel(
            num_sensors=num_sensors,
            num_horizons=3,
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            transformer_layers=config['model']['transformer_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            bilstm_hidden=config['model']['bilstm_hidden'],
            bilstm_layers=config['model']['bilstm_layers'],
            dropout=config['model']['dropout']
        ),
        # Hyperparameter Ablations on the Proposed Hybrid
        'Ablation: 1 Transformer Layer': lambda: TransformerBiLSTMModel(
            num_sensors=num_sensors,
            num_horizons=3,
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            transformer_layers=1,  # Reduced layers
            dim_feedforward=config['model']['dim_feedforward'],
            bilstm_hidden=config['model']['bilstm_hidden'],
            bilstm_layers=config['model']['bilstm_layers'],
            dropout=config['model']['dropout']
        ),
        'Ablation: 2 Attention Heads': lambda: TransformerBiLSTMModel(
            num_sensors=num_sensors,
            num_horizons=3,
            d_model=config['model']['d_model'],
            num_heads=2,  # Reduced attention heads
            transformer_layers=config['model']['transformer_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            bilstm_hidden=config['model']['bilstm_hidden'],
            bilstm_layers=config['model']['bilstm_layers'],
            dropout=config['model']['dropout']
        ),
        'Ablation: BiLSTM Hidden Size = 32': lambda: TransformerBiLSTMModel(
            num_sensors=num_sensors,
            num_horizons=3,
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            transformer_layers=config['model']['transformer_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            bilstm_hidden=32,  # Reduced hidden size
            bilstm_layers=config['model']['bilstm_layers'],
            dropout=config['model']['dropout']
        ),
        'Ablation: Dropout = 0.0': lambda: TransformerBiLSTMModel(
            num_sensors=num_sensors,
            num_horizons=3,
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            transformer_layers=config['model']['transformer_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            bilstm_hidden=config['model']['bilstm_hidden'],
            bilstm_layers=config['model']['bilstm_layers'],
            dropout=0.0  # No dropout
        )
    }
    
    ablation_results = []
    
    for exp_name, model_fn in experiments.items():
        print("=" * 60)
        print(f"Running Ablation Experiment: {exp_name}")
        print("=" * 60)
        
        # Instantiate model
        model = model_fn()
        
        trainer = Trainer(
            model=model,
            device=config['experiment']['device'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            grad_clip=config['training']['grad_clip'],
            patience=config['training']['patience']
        )
        
        # We run core models for full training config epochs, hyperparameter ablations for reduced epochs
        epochs = config['training']['epochs'] if exp_name in ['BiLSTM Only', 'Transformer Only', 'Proposed Hybrid'] else ablation_epochs
        
        train_losses, val_losses = trainer.fit(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            epochs=epochs,
            batch_size=config['training']['batch_size'],
            model_name=f"ablation_{exp_name.replace(' ', '_').replace(':', '')}_{dataset}.pt"
        )
        
        # Predict on test set
        preds_norm = trainer.predict(X_test, batch_size=config['training']['batch_size'])
        
        # Denormalize
        preds_raw = np.zeros_like(preds_norm)
        targets_raw = np.zeros_like(Y_test)
        for h in range(3):
            preds_raw[:, h, :] = scaler.inverse_transform(preds_norm[:, h, :])
            targets_raw[:, h, :] = scaler.inverse_transform(Y_test[:, h, :])
            
        # Compute overall average metrics (mean over all 3 horizons)
        metrics = compute_all_metrics(targets_raw, preds_raw)
        print(f"Ablation '{exp_name}' | Overall MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | MAPE: {metrics['MAPE']:.4f}%")
        
        ablation_results.append({
            'Experiment': exp_name,
            'Overall_MAE': metrics['MAE'],
            'Overall_RMSE': metrics['RMSE'],
            'Overall_MAPE': metrics['MAPE'],
            'Overall_sMAPE': metrics['sMAPE']
        })
        
    # Save results to ablation_results.csv
    df_ablation = pd.DataFrame(ablation_results)
    os.makedirs('outputs/tables', exist_ok=True)
    csv_path = f"outputs/tables/ablation_results_{dataset}.csv"
    df_ablation.to_csv(csv_path, index=False)
    print(f"Ablation study results table saved at: {csv_path}")

if __name__ == '__main__':
    run_ablation_study()

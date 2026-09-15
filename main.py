import argparse
import os
import sys
import yaml
import numpy as np
import torch

from experiments.run_baselines import run_baselines
from experiments.run_hybrid import run_hybrid
from experiments.run_ablation import run_ablation_study
from visualization.plot_comparison import plot_model_comparison
from visualization.plot_predictions import plot_actual_vs_predicted, plot_error_distribution
from explainability.shap_analysis import SHAPAnalyzer
from models.transformer_bilstm import TransformerBiLSTMModel
from preprocessing.create_sequences import get_train_val_test_data

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_shap_experiment(dataset_name):
    config = load_config()
    if dataset_name:
        config['dataset']['name'] = dataset_name
        
    dataset = config['dataset']['name']
    raw_dir = config['dataset']['raw_dir']
    from preprocessing.load_data import resolve_dataset_path
    file_path = resolve_dataset_path(raw_dir, dataset)
    
    print(f"Loading data for SHAP analysis ({dataset})...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler = get_train_val_test_data(
        file_path=file_path,
        input_window=config['sequence']['input_window'],
        horizons=[3, 6, 12],
        train_ratio=config['experiment']['train_ratio'],
        val_ratio=config['experiment']['val_ratio'],
        test_ratio=config['experiment']['test_ratio']
    )
    
    num_sensors = X_train.shape[2]
    
    # Load trained model checkpoint
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
    
    ckpt_path = f"checkpoints/hybrid_{dataset}.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: Trained hybrid model checkpoint not found at {ckpt_path}.")
        print("Please train the hybrid model first using: python main.py --model hybrid")
        return
        
    print(f"Loading model checkpoint from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    
    # Initialize analyzer (runs on CPU for Kernel SHAP stability/compatibility)
    analyzer = SHAPAnalyzer(model=model, background_data=X_train, device='cpu')
    
    # Explain predictions for the first sensor (index 0) and the 15m horizon (index 0)
    # horizons index: 0 = 15m, 1 = 30m, 2 = 60m
    shap_vals, test_sub = analyzer.explain_prediction(X_test, target_sensor=0, target_horizon_idx=0)
    
    # Generate and save plots
    analyzer.plot_and_save_shap(
        shap_values=shap_vals,
        test_subset=test_sub,
        target_sensor_name="Sensor 773869" if dataset == "METR-LA" else ("Sensor 400001" if dataset == "PEMS-BAY" else "Sensor 0"),
        horizon_min=15
    )
    print("SHAP explanation plots successfully generated under: outputs/figures/")

def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Traffic Flow Prediction Using Hybrid Transformer-BiLSTM and Explainable AI"
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='METR-LA',
        help="Target dataset name (default: METR-LA)"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['lstm', 'gru', 'bilstm', 'transformer', 'arima', 'hybrid'], 
        help="Train and predict using a single model"
    )
    parser.add_argument(
        '--experiment', 
        type=str, 
        choices=['baselines', 'ablation', 'shap'], 
        help="Run specific experiment runs (baselines: evaluates all baselines, ablation: run ablation study, shap: run explainability)"
    )
    
    args = parser.parse_args()
    
    # Verify dataset exists before running
    config = load_config()
    raw_dir = config['dataset']['raw_dir']
    target_dir = os.path.join(raw_dir, args.dataset)
    file_path = None
    if os.path.exists(target_dir):
        files = os.listdir(target_dir)
        for f in files:
            if f.lower().endswith(('.h5', '.hdf5', '.npz', '.csv')):
                file_path = os.path.join(target_dir, f)
                break
                
    if file_path is None or not os.path.exists(file_path):
        print(f"Error: No valid dataset file found under: {target_dir}")
        print("Please ensure a valid dataset file (.h5, .npz, or .csv) exists in that folder.")
        sys.exit(1)
        
    if args.model:
        if args.model == 'hybrid':
            # Train and evaluate proposed model
            run_hybrid(dataset_name=args.dataset)
        else:
            # Train and evaluate baseline model
            run_baselines(dataset_name=args.dataset, model_to_run=args.model)
            
        # Draw actual vs predicted plots for standard models if predictions exist
        preds_path = f"outputs/predictions/{args.model}_{args.dataset}_preds.npy"
        targets_path = f"outputs/predictions/{args.model}_{args.dataset}_targets.npy"
        if os.path.exists(preds_path) and os.path.exists(targets_path):
            preds = np.load(preds_path)
            targets = np.load(targets_path)
            # Plot for first sensor (index 0), 15m horizon (index 0)
            plot_actual_vs_predicted(targets[:, 0, 0], preds[:, 0, 0], sensor_name="0", horizon_min=15, dataset_name=args.dataset)
            plot_error_distribution(targets[:, 0, 0], preds[:, 0, 0], model_name=args.model, dataset_name=args.dataset)
            
    elif args.experiment:
        if args.experiment == 'baselines':
            # Run all baseline models
            run_baselines(dataset_name=args.dataset)
            # Run proposed hybrid model to complete results comparison
            run_hybrid(dataset_name=args.dataset)
            # Plot comparison
            plot_model_comparison(f"outputs/tables/baseline_results_{args.dataset}.csv", dataset_name=args.dataset)
            
        elif args.experiment == 'ablation':
            # Run ablation studies
            run_ablation_study(dataset_name=args.dataset)
            
        elif args.experiment == 'shap':
            # Run SHAP explanations
            run_shap_experiment(dataset_name=args.dataset)
            
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

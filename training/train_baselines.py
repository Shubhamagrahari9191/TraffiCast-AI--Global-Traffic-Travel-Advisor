import os
import sys

# Add root folder to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_baselines import run_baselines

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Baseline Models")
    parser.add_argument('--dataset', type=str, default='METR-LA', help='Dataset name (e.g., METR-LA, PEMS-BAY, PEMS04)')
    parser.add_argument('--model', type=str, default=None, choices=['lstm', 'gru', 'bilstm', 'transformer', 'arima'])
    args = parser.parse_args()
    
    run_baselines(dataset_name=args.dataset, model_to_run=args.model)

if __name__ == '__main__':
    main()

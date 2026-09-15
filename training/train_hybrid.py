import os
import sys

# Add root folder to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_hybrid import run_hybrid

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train proposed Hybrid Model")
    parser.add_argument('--dataset', type=str, default='METR-LA', help='Dataset name (e.g., METR-LA, PEMS-BAY, PEMS04)')
    args = parser.parse_args()
    
    run_hybrid(dataset_name=args.dataset)

if __name__ == '__main__':
    main()

import os
import json

def create_notebook(filename, title, description, code_blocks):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    f"{description}"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    for cb in code_blocks:
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cb
        })
        
    os.makedirs("notebooks", exist_ok=True)
    filepath = os.path.join("notebooks", filename)
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Created notebook: {filepath}")

def main():
    # 01 Data Exploration
    create_notebook(
        "01_data_exploration.ipynb",
        "01. Exploratory Data Analysis",
        "This notebook loads and analyzes the raw METR-LA and PEMS-BAY traffic speed datasets.",
        [
            ["import sys\n", "sys.path.append('..')\n", "from preprocessing.load_data import load_raw_data\n", "import matplotlib.pyplot as plt"],
            ["# Load METR-LA\n", "df_la, idx_la = load_raw_data('../data/raw/METR-LA/metr-la.h5')\n", "print('METR-LA shape:', df_la.shape)\n", "df_la.head()"],
            ["# Plot some sensor velocities\n", "plt.figure(figsize=(12, 5))\n", "plt.plot(df_la.iloc[:288, 0], label='Sensor 0')\n", "plt.title('Traffic Speed in METR-LA (1 Day)')\n", "plt.xlabel('Time Step')\n", "plt.ylabel('Speed (mph)')\n", "plt.legend()\n", "plt.show()"]
        ]
    )
    
    # 02 Preprocessing
    create_notebook(
        "02_preprocessing.ipynb",
        "02. Data Preprocessing & Sequence Creation",
        "This notebook demonstrates cleaning (interpolation), scaling, and sliding window sequence generation.",
        [
            ["import sys\n", "sys.path.append('..')\n", "from preprocessing.create_sequences import get_train_val_test_data\n", "import yaml"],
            ["with open('../configs/config.yaml', 'r') as f:\n", "    config = yaml.safe_load(f)\n", "print(config)"],
            ["X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler = get_train_val_test_data(\n", "    '../data/raw/METR-LA/metr-la.h5',\n", "    input_window=12,\n", "    horizons=[3, 6, 12]\n", ")\n", "print('X_train shape:', X_train.shape)\n", "print('Y_train shape:', Y_train.shape)"]
        ]
    )
    
    # 03 Baseline Models
    create_notebook(
        "03_baseline_models.ipynb",
        "03. Baseline Model Training & Evaluation",
        "This notebook trains and evaluates the baseline models: ARIMA, LSTM, GRU, BiLSTM, and Transformer.",
        [
            ["import os\n", "os.chdir('..') # move to project root to run CLI commands\n", "import sys\n", "sys.path.append('.')"],
            ["# Train a baseline model via command-line interface\n", "!python main.py --dataset METR-LA --model lstm"],
            ["# Evaluate all baselines\n", "!python main.py --dataset METR-LA --experiment baselines"]
        ]
    )
    
    # 04 Hybrid Model
    create_notebook(
        "04_hybrid_model.ipynb",
        "04. Hybrid Transformer-BiLSTM Model Training",
        "This notebook trains the proposed Hybrid model and compares its results against the baselines.",
        [
            ["import os\n", "os.chdir('..')"],
            ["# Train hybrid model\n", "!python main.py --dataset METR-LA --model hybrid"],
            ["# View results comparison table\n", "import pandas as pd\n", "df = pd.read_csv('outputs/tables/baseline_results_METR-LA.csv')\n", "df"]
        ]
    )
    
    # 05 SHAP Analysis
    create_notebook(
        "05_shap_analysis.ipynb",
        "05. SHAP Explainability Analysis",
        "This notebook runs Kernel SHAP to compute and display spatiotemporal feature contributions.",
        [
            ["import os\n", "os.chdir('..')"],
            ["# Run SHAP analysis\n", "!python main.py --dataset METR-LA --experiment shap"],
            ["# Display generated SHAP figure\n", "from IPython.display import Image\n", "Image(filename='outputs/figures/shap_summary_plot_15m.png')"]
        ]
    )

if __name__ == '__main__':
    main()

import os
import sys
import numpy as np
import pandas as pd
import yaml
from flask import Flask, jsonify, render_template, request

# Add root folder to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.load_data import load_raw_data

app = Flask(__name__, template_folder='templates', static_folder='static')

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

# Cache loaded dataset metadata to avoid expensive disk operations
DATA_CACHE = {}

def get_dataset_metadata(dataset_name):
    if dataset_name in DATA_CACHE:
        return DATA_CACHE[dataset_name]
        
    config = load_config()
    raw_dir = config['dataset']['raw_dir']
    from preprocessing.load_data import resolve_dataset_path
    try:
        file_path = resolve_dataset_path(raw_dir, dataset_name)
        df, timestamps = load_raw_data(file_path)
        sensor_ids = list(df.columns)
        
        if timestamps is None:
            # Fallback to dummy 5-minute interval timestamps
            timestamps_index = pd.date_range(start='2018-01-01 00:00', periods=len(df), freq='5min')
            timestamps_str = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_index]
        else:
            timestamps_str = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps]
        
        # Calculate test split starting index based on train_ratio + val_ratio
        num_timestamps = len(df)
        train_ratio = config['experiment']['train_ratio']
        val_ratio = config['experiment']['val_ratio']
        val_end = int(num_timestamps * (train_ratio + val_ratio))
        
        metadata = {
            'sensor_ids': sensor_ids,
            'timestamps': timestamps_str,
            'test_start_idx': val_end
        }
        DATA_CACHE[dataset_name] = metadata
        return metadata
    except Exception as e:
        print(f"Error loading metadata for {dataset_name}: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/config')
def get_config():
    dataset = request.args.get('dataset', 'METR-LA')
    meta = get_dataset_metadata(dataset)
    if not meta:
        return jsonify({'error': f'Failed to load dataset: {dataset}'}), 500
        
    return jsonify({
        'dataset': dataset,
        'sensors': meta['sensor_ids'],
        'horizons': ['15 min', '30 min', '60 min']
    })

@app.route('/api/predictions')
def get_predictions():
    dataset = request.args.get('dataset', 'METR-LA')
    sensor_idx = int(request.args.get('sensor', '0'))
    horizon_idx = int(request.args.get('horizon', '0'))  # 0: 15m, 1: 30m, 2: 60m
    
    meta = get_dataset_metadata(dataset)
    if not meta:
        return jsonify({'error': 'Failed to load dataset metadata'}), 500
        
    sensor_ids = meta['sensor_ids']
    if sensor_idx < 0 or sensor_idx >= len(sensor_ids):
        return jsonify({'error': 'Invalid sensor index'}), 400
        
    # Mapping horizon index to step offset (15m = 3 steps, 30m = 6 steps, 60m = 12 steps)
    horizon_steps = [3, 6, 12]
    h_step = horizon_steps[horizon_idx]
    
    # Load targets (actual values)
    targets_path = f"outputs/predictions/hybrid_{dataset}_targets.npy"
    if not os.path.exists(targets_path):
        return jsonify({'error': f'Predictions for {dataset} not found. Please train models first.'}), 404
        
    targets = np.load(targets_path)  # shape: (num_samples, 3, num_sensors)
    num_samples = targets.shape[0]
    
    actual_vals = targets[:, horizon_idx, sensor_idx].tolist()
    
    # Slice the corresponding test timestamps
    input_window = 12
    test_start = meta['test_start_idx']
    
    timestamps_slice = []
    for i in range(num_samples):
        t_idx = test_start + i + input_window + h_step - 1
        if t_idx < len(meta['timestamps']):
            timestamps_slice.append(meta['timestamps'][t_idx])
        else:
            timestamps_slice.append(f"Step {i}")
            
    # Load predictions for all baselines and hybrid
    models = ['hybrid', 'lstm', 'gru', 'bilstm', 'transformer', 'arima']
    model_preds = {}
    config = load_config()
    
    for model in models:
        preds_path = f"outputs/predictions/{model}_{dataset}_preds.npy"
        if os.path.exists(preds_path):
            preds = np.load(preds_path)
            
            if model == 'arima':
                # ARIMA runs with a step size sampling (default: 50)
                step_size = 50 if not config['arima']['full_experiment'] else 200
                aligned_arima = [None] * num_samples
                
                # Check if this sensor index was actually evaluated in ARIMA config
                arima_sensors = config['arima']['selected_sensors']
                if config['arima']['full_experiment']:
                    arima_sensors = list(range(len(sensor_ids)))
                    
                if sensor_idx in arima_sensors:
                    arima_s_idx = arima_sensors.index(sensor_idx)
                    for idx_out, i in enumerate(range(0, num_samples, step_size)):
                        if idx_out < preds.shape[0]:
                            aligned_arima[i] = float(preds[idx_out, horizon_idx, arima_s_idx])
                model_preds['arima'] = aligned_arima
            else:
                model_preds[model] = preds[:, horizon_idx, sensor_idx].tolist()
        else:
            model_preds[model] = [None] * num_samples
            
    return jsonify({
        'sensor_id': sensor_ids[sensor_idx],
        'sensor_idx': sensor_idx,
        'timestamps': timestamps_slice,
        'actual': actual_vals,
        'predictions': model_preds
    })

@app.route('/api/metrics')
def get_metrics():
    dataset = request.args.get('dataset', 'METR-LA')
    results_path = f"outputs/tables/baseline_results_{dataset}.csv"
    significance_path = f"outputs/tables/statistical_significance_{dataset}.csv"
    
    metrics = []
    significance = []
    
    if os.path.exists(results_path):
        df_m = pd.read_csv(results_path)
        metrics = df_m.to_dict(orient='records')
        
    if os.path.exists(significance_path):
        df_s = pd.read_csv(significance_path)
        significance = df_s.to_dict(orient='records')
        
    return jsonify({
        'metrics': metrics,
        'significance': significance
    })

@app.route('/api/ablation')
def get_ablation():
    dataset = request.args.get('dataset', 'METR-LA')
    ablation_path = f"outputs/tables/ablation_results_{dataset}.csv"
    
    results = []
    if os.path.exists(ablation_path):
        df_a = pd.read_csv(ablation_path)
        results = df_a.to_dict(orient='records')
        
    return jsonify({
        'ablation': results
    })

# ==========================================
# Global Location & Travel Decision API Routes
# ==========================================
from services.global_traffic import geocode_address, plan_global_route, POPULAR_CITIES

@app.route('/api/global/popular-cities')
def get_popular_cities():
    """Return pre-configured popular global cities."""
    return jsonify({
        'cities': POPULAR_CITIES
    })

@app.route('/api/global/geocode')
def api_geocode():
    """Geocode any global location or address string."""
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'results': []})
    results = geocode_address(q)
    return jsonify({'results': results})

@app.route('/api/global/route')
def api_global_route():
    """Calculate worldwide route, 24h traffic congestion forecast, and departure recommendations."""
    try:
        origin_lat = float(request.args.get('origin_lat', '40.7128'))
        origin_lng = float(request.args.get('origin_lng', '-74.0060'))
        dest_lat = float(request.args.get('dest_lat', '40.6413'))
        dest_lng = float(request.args.get('dest_lng', '-73.7781'))
        origin_name = request.args.get('origin_name', 'Origin')
        dest_name = request.args.get('dest_name', 'Destination')
        day_type = request.args.get('day_type', 'weekday')

        route_data = plan_global_route(
            origin_lat, origin_lng, dest_lat, dest_lng,
            origin_name=origin_name, dest_name=dest_name,
            day_type=day_type
        )
        return jsonify(route_data)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/outputs/figures/<path:filename>')
def serve_figure(filename):
    from flask import send_from_directory
    return send_from_directory('outputs/figures', filename)

if __name__ == '__main__':
    # Running local development server on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)

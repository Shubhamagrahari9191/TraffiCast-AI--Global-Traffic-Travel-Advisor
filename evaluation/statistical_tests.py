import numpy as np
from scipy import stats

def perform_wilcoxon_test(
    y_true: np.ndarray, 
    y_pred_baseline: np.ndarray, 
    y_pred_proposed: np.ndarray,
    max_samples: int = 10000,
    seed: int = 42
) -> dict:
    """
    Performs the Wilcoxon signed-rank test on paired prediction errors to evaluate
    whether the proposed hybrid model achieves systematically lower error than a baseline model.
    
    Statistical Hypotheses:
    - Null Hypothesis (H0): The median of the paired absolute error differences
      (D = |y_true - y_pred_baseline| - |y_true - y_pred_proposed|) is less than or equal to zero.
      (i.e., The proposed model does NOT produce systematically smaller errors).
    - Alternative Hypothesis (H1): The median of the paired absolute error differences D is greater than zero
      (one-sided test, alternative='greater').
      (i.e., The proposed model's prediction errors are systematically smaller than the baseline's errors).
      
    Note on Dependence & Pairing:
    - Observations are strictly paired: each baseline error is paired with the proposed model error
      evaluated on the EXACT SAME timestamp, sensor, and forecast horizon.
    - Slices for distinct forecast horizons must be tested separately (not mixed together).
    - To maintain computational feasibility and avoid ranking tie degeneration over millions of grid points,
      a reproducible paired subsample (default N=10,000, seed=42) is drawn without replacement.
    
    Args:
        y_true (np.ndarray): Ground truth target array (num_samples, num_sensors).
        y_pred_baseline (np.ndarray): Predictions from baseline model, matching y_true shape.
        y_pred_proposed (np.ndarray): Predictions from proposed model, matching y_true shape.
        max_samples (int): Maximum paired samples to evaluate in Scipy.
        seed (int): Fixed random seed for reproducible subsampling.
        
    Returns:
        dict: Test results containing statistic, p-value, significance flag, and hypothesis descriptions.
    """
    # Verify shape consistency
    assert y_true.shape == y_pred_baseline.shape == y_pred_proposed.shape, (
        f"Shape mismatch: y_true={y_true.shape}, baseline={y_pred_baseline.shape}, proposed={y_pred_proposed.shape}"
    )
    
    # Calculate sample-wise paired absolute errors
    errors_baseline = np.abs(y_true - y_pred_baseline).flatten()
    errors_proposed = np.abs(y_true - y_pred_proposed).flatten()
    
    # Filter valid non-NaN pairs
    valid_mask = ~np.isnan(errors_baseline) & ~np.isnan(errors_proposed)
    errors_baseline = errors_baseline[valid_mask]
    errors_proposed = errors_proposed[valid_mask]
    
    total_paired_observations = len(errors_baseline)
    subsampled = False
    
    if total_paired_observations > max_samples:
        rng = np.random.default_rng(seed=seed)
        sample_indices = rng.choice(total_paired_observations, size=max_samples, replace=False)
        errors_baseline = errors_baseline[sample_indices]
        errors_proposed = errors_proposed[sample_indices]
        subsampled = True
        
    # Wilcoxon signed-rank test (one-sided: alternative='greater' tests errors_baseline > errors_proposed)
    try:
        # Zero-differences (ties) are discarded by standard Wilcoxon procedure ('wilcox' method in scipy)
        res = stats.wilcoxon(errors_baseline, errors_proposed, alternative='greater')
        statistic = float(res.statistic)
        p_value = float(res.pvalue)
        is_significant = bool(p_value < 0.05)
    except Exception as e:
        statistic, p_value, is_significant = float('nan'), float('nan'), False
        print(f"Wilcoxon signed-rank test could not be computed: {e}")
        
    interpretation = (
        f"Reject H0 (p = {p_value:.2e} < 0.05). The paired error differences indicate that "
        f"the Proposed Hybrid model achieves systematically smaller prediction errors than the baseline."
        if is_significant else
        f"Fail to reject H0 (p = {p_value:.4f} >= 0.05). No statistically significant difference "
        f"in paired prediction errors at alpha=0.05."
    )
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'is_significant': is_significant,
        'interpretation': interpretation,
        'subsampled': subsampled,
        'evaluated_sample_size': len(errors_baseline),
        'total_available_pairs': total_paired_observations,
        'null_hypothesis': "Median difference (Baseline Error - Proposed Error) <= 0",
        'alternative_hypothesis': "Median difference (Baseline Error - Proposed Error) > 0 (one-sided)"
    }

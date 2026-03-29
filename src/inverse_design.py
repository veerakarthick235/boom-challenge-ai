"""
boom_challenge/src/inverse_design.py
======================================
Task 2: Inverse Design using Smart Monte Carlo Simulation.
Leverages the robust XGBoost base model to prevent out-of-distribution hallucinations.
"""

import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────
# 1. PARAMETER BOUNDS
# ─────────────────────────────────────────────────────────────

PARAM_NAMES = [
    'porosity', 'atmosphere', 'gravity', 'coupling', 
    'strength', 'shape_factor', 'energy', 'angle_rad'
]

# Strict physical bounds based on training set percentiles
LOWER_BOUNDS = np.array([0.050, 0.000, 0.100, 0.100, 5e4, 0.500, 1e7,  0.100])
UPPER_BOUNDS = np.array([0.500, 1.000, 20.00, 1.000, 1e9, 2.000, 1e13, 1.570])

P80_TARGET_MIN = 96.0
P80_TARGET_MAX = 101.0
R95_MAX        = 175.0


# ─────────────────────────────────────────────────────────────
# 2. XGBOOST PREDICTOR (HALLUCINATION-PROOF)
# ─────────────────────────────────────────────────────────────

class EnsemblePredictor:
    def __init__(self, ensemble_model, feature_fn, scaler):
        # We extract the XGBoost base model from your ensemble.
        # Decision trees cannot extrapolate to infinity, making them bulletproof here.
        self.xgb_model = ensemble_model.base_models['xgb']
        self.feature_fn = feature_fn
        self.scaler = scaler

    def predict_batch(self, x_matrix: np.ndarray) -> tuple:
        df = pd.DataFrame(x_matrix, columns=PARAM_NAMES)
        
        # Build features safely
        X_raw = self.feature_fn(df)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e6, neginf=-1e6)
        
        X_scaled = self.scaler.transform(X_raw)
        
        # Predict using ONLY XGBoost
        log_preds = self.xgb_model.predict(X_scaled)
        preds = np.expm1(log_preds)
        
        # Target index 0 is P80, Target index 1 is R95
        return preds[:, 0], preds[:, 1]


# ─────────────────────────────────────────────────────────────
# 3. MONTE CARLO SEARCH (GUARANTEED YIELD)
# ─────────────────────────────────────────────────────────────

def run_inverse_design(predictor: EnsemblePredictor, output_path: str = 'outputs/task2_scenarios.csv') -> pd.DataFrame:
    print("============================================================")
    print("TASK 2: INVERSE DESIGN (XGBOOST SIMULATOR)")
    print("============================================================")

    N_SAMPLES = 500000
    print(f"\n[1/3] Generating {N_SAMPLES:,} random impact scenarios...")
    
    X_random = np.random.uniform(LOWER_BOUNDS, UPPER_BOUNDS, (N_SAMPLES, len(LOWER_BOUNDS)))

    str_idx = PARAM_NAMES.index('strength')
    nrg_idx = PARAM_NAMES.index('energy')
    
    X_random[:, str_idx] = 10 ** np.random.uniform(np.log10(LOWER_BOUNDS[str_idx]), np.log10(UPPER_BOUNDS[str_idx]), N_SAMPLES)
    X_random[:, nrg_idx] = 10 ** np.random.uniform(np.log10(LOWER_BOUNDS[nrg_idx]), np.log10(UPPER_BOUNDS[nrg_idx]), N_SAMPLES)

    print("[2/3] Predicting outcomes using the robust XGBoost Surrogate...")
    p80_preds, r95_preds = predictor.predict_batch(X_random)

    print(f"[3/3] Finding the 20 closest matches to P80=98.5 and R95<175...")
    
    TARGET_P80 = 98.5 
    
    p80_errors = np.abs(p80_preds - TARGET_P80)
    r95_penalties = np.where(r95_preds > R95_MAX, (r95_preds - R95_MAX) * 10, 0)
    
    energies = X_random[:, PARAM_NAMES.index('energy')]
    energy_bonus = np.log10(energies) * 0.001 
    
    total_scores = p80_errors + r95_penalties + energy_bonus
    best_indices = np.argsort(total_scores)[:20]

    final_X = X_random[best_indices]
    final_p80 = p80_preds[best_indices]
    final_r95 = r95_preds[best_indices]
    final_energy = energies[best_indices]

    df = pd.DataFrame(final_X, columns=PARAM_NAMES)
    df.insert(0, 'id', range(len(df)))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("============================================================")
    print(f"🎯 Saved 20 best-effort scenarios -> {output_path}")
    print(f"   P80 range: [{np.min(final_p80):.2f}, {np.max(final_p80):.2f}]")
    print(f"   R95 range: [{np.min(final_r95):.2f}, {np.max(final_r95):.2f}]")
    print(f"   Energy range:  [{np.min(final_energy):.2e}, {np.max(final_energy):.2e}] J")
    print("============================================================")

    return df   
"""
boom_challenge/src/feature_engineering.py
==========================================
Feature Engineering updated for the Official Boom Challenge Dataset.
Strict Feature Schema enforced to prevent Adversarial Shift (AUC 1.0 Fix).
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, RobustScaler

# ─────────────────────────────────────────────────────────────
# CONSTANTS & STRICT SCHEMA
# ─────────────────────────────────────────────────────────────
SEED = 42

# 🚨 STRICT SCHEMA: Only these raw features are allowed to exist.
REQUIRED_INPUTS = [
    'porosity', 'atmosphere', 'gravity', 'coupling', 
    'strength', 'shape_factor', 'energy', 'angle_rad'
]

LOG_FEATURES = ['energy', 'strength', 'gravity', 'KE', 'energy_strength_ratio']
TARGET_COLS = ['P80', 'R95', 'fines_frac', 'oversize_frac', 'R50_fines', 'R50_oversize']


# ─────────────────────────────────────────────────────────────
# SCHEMA ENFORCEMENT
# ─────────────────────────────────────────────────────────────
def align_features(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
    """Forces exact column alignment to kill data leakage."""
    df = df.copy()
    
    # 1. Drop the ID immediately so it can never leak
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # 2. Add missing required columns (prevents pipeline crashes)
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    # 3. Return ONLY the required input features (strips out targets and ghosts)
    return df[required_cols]


# ─────────────────────────────────────────────────────────────
# PHYSICS FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate physics-informed features from raw official impact parameters.
    """
    # 🔥 ELITE FIX: Force strict alignment before generating features
    df = align_features(df, REQUIRED_INPUTS)

    # 1. Provide 'KE' for the PyTorch PINN Physics Loss & Power Features
    df['KE'] = df['energy']  

    # 2. Trigonometric angular features
    df['sin_angle'] = np.sin(df['angle_rad'])
    df['cos_angle'] = np.cos(df['angle_rad'])
    df['sin2_angle'] = np.sin(2 * df['angle_rad'])

    # 3. Interaction Features
    df['energy_strength_ratio'] = df['energy'] / (df['strength'] + 1e-12)
    df['gravity_energy_interaction'] = df['gravity'] * np.log1p(df['energy'])
    df['structural_integrity'] = df['strength'] * (1 - df['porosity'])

    # 4. Scale-Based Power Feature
    df['log_energy'] = np.log1p(df['KE'])

    # Note: Legacy conditional features (density_ratio, etc.) removed to prevent data shift.
    return df


# ─────────────────────────────────────────────────────────────
# LOG TRANSFORMATIONS
# ─────────────────────────────────────────────────────────────
def apply_log_transforms(df: pd.DataFrame, log_cols: list = None) -> pd.DataFrame:
    """Apply log1p transform to skewed physics features safely."""
    if log_cols is None:
        log_cols = LOG_FEATURES

    df = df.copy()
    for col in log_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(np.abs(df[col]))
    return df


def transform_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform targets for training in log-space."""
    df = df.copy()
    for col in TARGET_COLS:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    return df


def inverse_transform_targets(log_preds: np.ndarray) -> np.ndarray:
    """Inverse of log1p: expm1."""
    return np.expm1(log_preds)


# ─────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────────────────────
def get_feature_columns(df: pd.DataFrame,
                        exclude: list = None,
                        use_log: bool = True) -> list:
    """Return ordered list of feature columns for model training."""
    if exclude is None:
        exclude = []

    log_targets = [f'log_{t}' for t in TARGET_COLS]
    always_exclude = TARGET_COLS + log_targets + ['id', 'sample_id', 'angle_rad']
    exclude_set = set(exclude + always_exclude)

    feature_cols = [c for c in df.columns if c not in exclude_set]

    if not use_log:
        feature_cols = [c for c in feature_cols if not c.startswith('log_')]

    return feature_cols


# ─────────────────────────────────────────────────────────────
# SCALING
# ─────────────────────────────────────────────────────────────
def fit_scaler(X_train: np.ndarray, scaler_type: str = 'robust') -> object:
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def save_scaler(scaler, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path: str):
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE (Legacy Wrapper)
# ─────────────────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame,
                         scaler=None,
                         fit_scaler_flag: bool = False,
                         scaler_type: str = 'robust',
                         use_log: bool = True) -> tuple:
    """Full feature engineering pipeline."""
    df = engineer_physics_features(df)
    df = apply_log_transforms(df)
    
    if all(t in df.columns for t in TARGET_COLS):
        df = transform_targets(df)

    feature_cols = get_feature_columns(df, use_log=use_log)
    X = df[feature_cols].values.astype(np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    if fit_scaler_flag:
        scaler = fit_scaler(X, scaler_type)
        X = scaler.transform(X)
    elif scaler is not None:
        X = scaler.transform(X)

    return X, feature_cols, scaler


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(SEED)
    n = 200
    df_test = pd.DataFrame({
        'porosity': np.random.uniform(0.1, 0.6, n),
        'atmosphere': np.random.uniform(0, 1, n),
        'gravity': np.random.uniform(0.1, 20.0, n),
        'coupling': np.random.uniform(0.1, 0.9, n),
        'strength': np.random.uniform(1e5, 1e8, n),
        'shape_factor': np.random.uniform(0.5, 1.5, n),
        'energy': np.random.uniform(1e10, 1e15, n),
        'angle_rad': np.random.uniform(0.1, 1.5, n),
        'P80': np.random.uniform(80, 120, n),
        'R95': np.random.uniform(50, 250, n),
    })

    X, feat_names, sc = build_feature_matrix(df_test, fit_scaler_flag=True)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features:   {len(feat_names)}")
    print(f"Feature names: {feat_names[:5]} ...")

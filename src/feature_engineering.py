"""
boom_challenge/src/feature_engineering.py
==========================================
Physics-Informed Feature Engineering for the Boom: Trajectory Unknown Challenge.

Implements:
  - Buckingham π dimensionless groups
  - Housen-Holsapple impact scaling features
  - Kinematic and energetic features
  - Log-space transformations
  - Target engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
SEED = 42
LOG_FEATURES = [
    'v_i', 'm_i', 'rho_i', 'rho_t', 'Y', 'g', 'd_i',
    'KE', 'momentum', 'E_spec', 'pi_v', 'pi_2',
    'mu_crater', 'KE_per_area', 'mass_over_strength'
]

TARGET_COLS = ['P80', 'R95']


# ─────────────────────────────────────────────────────────────
# PHYSICS FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate physics-informed features from raw impact parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: v_i, m_i, rho_i, rho_t, Y, g, d_i, theta

    Returns
    -------
    pd.DataFrame with original + engineered features
    """
    df = df.copy()

    # ── Kinematic Features ────────────────────────────────────
    df['KE'] = 0.5 * df['m_i'] * df['v_i'] ** 2          # Kinetic energy (J)
    df['momentum'] = df['m_i'] * df['v_i']                # Linear momentum (kg·m/s)
    df['v_i_sq'] = df['v_i'] ** 2                         # Velocity squared
    df['impactor_volume'] = (np.pi / 6) * df['d_i'] ** 3  # Sphere volume (m³)

    # ── Energetic Scaling Features ────────────────────────────
    # Specific impact energy (energy / target material volume)
    target_vol_proxy = df['rho_t'] * df['d_i'] ** 3
    df['E_spec'] = df['KE'] / (target_vol_proxy + 1e-12)

    # Energy per unit area (cratering pressure proxy)
    df['KE_per_area'] = df['KE'] / (np.pi / 4 * df['d_i'] ** 2 + 1e-12)

    # ── Buckingham π Groups (Dimensionless) ───────────────────
    # π₁: Velocity shock ratio (Housen-Holsapple)
    df['pi_v'] = df['v_i'] / (np.sqrt(df['Y'] / (df['rho_t'] + 1e-12)) + 1e-12)

    # π₂: Gravity-inertia transition parameter
    df['pi_2'] = (df['rho_t'] * df['g'] * df['d_i']) / (df['Y'] + 1e-12)

    # π₃: Density ratio (impactor/target)
    df['density_ratio'] = df['rho_i'] / (df['rho_t'] + 1e-12)

    # π₄: Gravity-to-kinetic ratio
    df['pi_grav'] = (df['g'] * df['d_i']) / (df['v_i'] ** 2 + 1e-12)

    # ── Cratering Efficiency Proxy ────────────────────────────
    df['mu_crater'] = (
        (df['KE'] / (df['Y'] + 1e-12)) ** (1 / 3) *
        (df['rho_i'] / (df['rho_t'] + 1e-12)) ** (1 / 3)
    )

    # ── Material / Structural Features ───────────────────────
    df['mass_over_strength'] = df['m_i'] / (df['Y'] * df['d_i'] ** 2 + 1e-12)
    df['impedance'] = df['rho_i'] * df['v_i']             # Shock impedance (kg/m²/s)

    # ── Geometric / Angular Features ──────────────────────────
    df['theta_rad'] = np.radians(df['theta'])
    df['sin_theta'] = np.sin(df['theta_rad'])              # Obliquity factor
    df['cos_theta'] = np.cos(df['theta_rad'])
    df['sin2_theta'] = np.sin(2 * df['theta_rad'])         # Optimal angle proxy

    # ── Interaction Features ─────────────────────────────────
    df['log_v_log_m'] = np.log1p(df['v_i']) * np.log1p(df['m_i'])
    df['log_KE_sin'] = np.log1p(df['KE']) * df['sin_theta']
    df['v_density_product'] = df['v_i'] * df['rho_i']
    df['KE_gravity'] = df['KE'] / (df['g'] * df['m_i'] + 1e-12)

    return df


# ─────────────────────────────────────────────────────────────
# LOG TRANSFORMATIONS
# ─────────────────────────────────────────────────────────────

def apply_log_transforms(df: pd.DataFrame, log_cols: list = None) -> pd.DataFrame:
    """
    Apply log1p transform to skewed physics features.
    Handles zeros safely with log1p.
    """
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
    """
    Return ordered list of feature columns for model training.

    Parameters
    ----------
    df : pd.DataFrame
    exclude : list of columns to exclude (e.g., ID cols, targets)
    use_log : bool — whether to include log-transformed features
    """
    if exclude is None:
        exclude = []

    always_exclude = TARGET_COLS + ['log_P80', 'log_R95',
                                    'theta_rad', 'id', 'sample_id']
    exclude_set = set(exclude + always_exclude)

    feature_cols = [c for c in df.columns if c not in exclude_set]

    if not use_log:
        feature_cols = [c for c in feature_cols if not c.startswith('log_')]

    return feature_cols


# ─────────────────────────────────────────────────────────────
# SCALING
# ─────────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray, scaler_type: str = 'robust') -> object:
    """Fit and return a feature scaler."""
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler, path: str):
    """Persist scaler to disk for inference."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str):
    """Load persisted scaler."""
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame,
                         scaler=None,
                         fit_scaler_flag: bool = False,
                         scaler_type: str = 'robust',
                         use_log: bool = True
                         ) -> tuple:
    """
    Full feature engineering pipeline.

    Returns (X_array, feature_names, scaler)
    """
    df = engineer_physics_features(df)
    df = apply_log_transforms(df)
    df = transform_targets(df)

    feature_cols = get_feature_columns(df, use_log=use_log)
    X = df[feature_cols].values.astype(np.float32)

    # Handle NaN / Inf
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
    # Synthetic data test
    np.random.seed(SEED)
    n = 200
    df_test = pd.DataFrame({
        'v_i':   np.random.uniform(1e3, 2e4, n),   # m/s
        'm_i':   np.random.uniform(1e8, 1e12, n),   # kg
        'rho_i': np.random.uniform(1.5, 8.0, n),    # g/cm³
        'rho_t': np.random.uniform(1.5, 3.5, n),
        'Y':     np.random.uniform(1e6, 1e9, n),    # Pa
        'g':     np.random.uniform(0.1, 15.0, n),   # m/s²
        'd_i':   np.random.uniform(10, 5000, n),    # m
        'theta': np.random.uniform(10, 80, n),      # degrees
        'P80':   np.random.uniform(80, 120, n),
        'R95':   np.random.uniform(50, 250, n),
    })

    X, feat_names, sc = build_feature_matrix(df_test, fit_scaler_flag=True)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features:   {len(feat_names)}")
    print(f"Feature names: {feat_names[:10]} ...")

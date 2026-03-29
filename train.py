"""
Main training pipeline for the Boom: Trajectory Unknown Challenge.

Orchestrates:
  1. Data loading and EDA
  2. Physics feature engineering
  3. XGBoost training with Optuna HPO
  4. LightGBM training with Optuna HPO
  5. PINN training (PyTorch)
  6. Stacking ensemble
  7. Test set predictions (Task 1)
  8. Inverse design (Task 2)
  9. Output CSV generation

Usage:
    python train.py --data_dir data/ --output_dir outputs/
                    --run_hpo --run_task2
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import mlflow

from src.feature_engineering import build_feature_matrix, inverse_transform_targets
from src.models.gbm_models import XGBModel, LGBMModel, tune_xgboost, tune_lightgbm
from src.models.pinn_model import PINNTrainer
from src.ensemble import (StackingEnsemble, generate_oof_predictions,
                           adversarial_validation, evaluate_predictions)
from src.inverse_design import run_inverse_design, EnsemblePredictor

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────

SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> tuple:
    """Load train, train_labels, and test CSV files."""
    train_features_path = os.path.join(data_dir, 'train.csv')
    train_labels_path = os.path.join(data_dir, 'train_labels.csv')
    test_path  = os.path.join(data_dir, 'test.csv')

    train_features = pd.read_csv(train_features_path)
    train_labels = pd.read_csv(train_labels_path)
    test_df  = pd.read_csv(test_path)

    # 🛠️ THE FIX: Check if 'id' exists in the labels file.
    if 'id' in train_labels.columns:
        train_df = pd.merge(train_features, train_labels, on='id')
    else:
        # If no 'id' column, just glue them side-by-side (row 1 matches row 1)
        train_df = pd.concat([train_features, train_labels], axis=1)

    # Safety net: Ensure test_df has an 'id' column for the final submission file
    if 'id' not in test_df.columns:
        test_df['id'] = range(len(test_df))

    print(f"Training data:  {train_df.shape}")
    print(f"Test data:      {test_df.shape}")
    
    target_cols = ['P80', 'R95', 'fines_frac', 'oversize_frac', 'R50_fines', 'R50_oversize']
    print(f"\nTarget stats:\n{train_df[target_cols].describe()}")

    return train_df, test_df


# ─────────────────────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame, output_dir: str):
    """Generate EDA visualizations."""
    os.makedirs(os.path.join(output_dir, 'eda'), exist_ok=True)

    # Target distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df['P80'].hist(bins=50, ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('P80 Distribution'); axes[0].set_xlabel('P80 (m)')
    df['R95'].hist(bins=50, ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_title('R95 Distribution'); axes[1].set_xlabel('R95 (m)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda', 'target_distributions.png'), dpi=150)
    plt.close()

    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax,
                linewidths=0.3)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda', 'correlation_heatmap.png'), dpi=150)
    plt.close()

    print("EDA plots saved.")


# ─────────────────────────────────────────────────────────────
# FEATURE MATRIX PREPARATION (🔥 ELITE UPGRADE)
# ─────────────────────────────────────────────────────────────

def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """STRICT Data prep to prevent Data Leakage and Adversarial Shift."""
    TARGET_COLS = ['P80', 'R95', 'fines_frac', 'oversize_frac', 'R50_fines', 'R50_oversize']
    TARGET_LOG = [f'log_{t}' for t in TARGET_COLS]

    from src.feature_engineering import engineer_physics_features, apply_log_transforms, get_feature_columns
    from sklearn.preprocessing import RobustScaler

    # 1. Engineer independently
    train_feat = apply_log_transforms(engineer_physics_features(train_df))
    test_feat = apply_log_transforms(engineer_physics_features(test_df))

    # 2. Force exact column alignment
    feat_names = get_feature_columns(train_feat)
    
    # Extract raw arrays
    X_train_raw = train_feat[feat_names].values.astype(np.float32)
    X_test_raw = test_feat[feat_names].values.astype(np.float32)

    # Clean extreme math errors
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=1e6, neginf=-1e6)

    # 3. STRICT ISOLATION SCALING (The AUC 1.0 Killer)
    scaler = RobustScaler()
    X_all = scaler.fit_transform(X_train_raw) # Fit ONLY on train
    X_test = scaler.transform(X_test_raw)     # Transform ONLY on test

    y_all = np.log1p(train_df[TARGET_COLS].values)

    X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.15, random_state=SEED)

    print(f"\nFeature matrix: {X_all.shape[1]} strictly aligned features")
    return {
        'X_all': X_all, 'y_all': y_all, 'X_tr': X_tr, 'y_tr': y_tr,
        'X_val': X_val, 'y_val': y_val, 'X_test': X_test,
        'feat_names': feat_names, 'scaler': scaler, 'target_log_names': TARGET_LOG
    }


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

def main(args):
    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("boom-trajectory-unknown")

    with mlflow.start_run(run_name=f"full_pipeline_v{args.version}"):

        # ── Load Data ──────────────────────────────────────────
        print("\n[1/8] Loading data...")
        train_df, test_df = load_data(args.data_dir)

        # ── EDA ───────────────────────────────────────────────
        print("\n[2/8] Running EDA...")
        run_eda(train_df, args.output_dir)

        # ── Feature Engineering ───────────────────────────────
        print("\n[3/8] Feature engineering...")
        data = prepare_features(train_df, test_df)
        X_all, y_all = data['X_all'], data['y_all']
        X_tr, y_tr   = data['X_tr'],  data['y_tr']
        X_val, y_val = data['X_val'], data['y_val']
        X_test       = data['X_test']
        scaler       = data['scaler']
        feat_names   = data['feat_names']
        target_log_names = data['target_log_names']

        # Adversarial validation
        adv = adversarial_validation(X_all, X_test, feat_names)
        mlflow.log_metric('adversarial_auc', adv['mean_auc'])

        # ── Hyperparameter Optimization ───────────────────────
        best_xgb_params, best_lgbm_params = {}, {}
        if args.run_hpo:
            print("\n[4/8] Optuna HPO (XGBoost)...")
            best_xgb_params = tune_xgboost(X_tr, y_tr[:, 0], n_trials=100)
            print("\n[4/8] Optuna HPO (LightGBM)...")
            best_lgbm_params = tune_lightgbm(X_tr, y_tr[:, 0], n_trials=100)
            mlflow.log_params({'xgb_best': str(best_xgb_params),
                               'lgbm_best': str(best_lgbm_params)})

        # ── OOF Generation for Stacking ───────────────────────
        print("\n[5/8] Generating OOF predictions (5-fold CV)...")

        # XGBoost OOF
        print("  XGBoost...")
        xgb_oof = generate_oof_predictions(
            XGBModel, {'params': best_xgb_params},
            X_all, y_all,
            fit_kwargs={'target_cols': target_log_names},
            model_name='XGBoost'
        )

        # LightGBM OOF
        print("  LightGBM...")
        lgbm_oof = generate_oof_predictions(
            LGBMModel, {'params': best_lgbm_params},
            X_all, y_all,
            fit_kwargs={'target_cols': target_log_names},
            model_name='LightGBM'
        )

        # PINN OOF
        print("  PINN (PyTorch)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ke_idx = feat_names.index('log_KE') if 'log_KE' in feat_names else 0
        pinn_oof = generate_oof_predictions(
            PINNTrainer,
            {'input_dim': X_all.shape[1], 'hidden_dims': [256, 256, 128, 64],
             'dropout': 0.15, 'lr': 5e-4, 'ke_feature_idx': ke_idx,
             'device': device},
            X_all, y_all,
            model_name='PINN'
        )

        # ── Train Final Base Models ───────────────────────────
        print("\n[6/8] Training final base models on full train set...")

        xgb_final  = XGBModel(best_xgb_params)
        lgbm_final = LGBMModel(best_lgbm_params)
        pinn_final = PINNTrainer(input_dim=X_all.shape[1],
                                  hidden_dims=[256, 256, 128, 64],
                                  dropout=0.15, lr=5e-4,
                                  ke_feature_idx=ke_idx, device=device)

        xgb_final.fit(X_tr, y_tr, X_val, y_val)
        lgbm_final.fit(X_tr, y_tr, X_val, y_val)
        pinn_final.fit(X_tr, y_tr, X_val, y_val, epochs=300)

        # Save models
        xgb_final.save(os.path.join(args.output_dir, 'models/xgb'))
        lgbm_final.save(os.path.join(args.output_dir, 'models/lgbm'))
        pinn_final.save(os.path.join(args.output_dir, 'models/pinn.pt'))

        # ── Stacking Ensemble ─────────────────────────────────
        print("\n[7/8] Generating elite ensemble predictions...")
        
        # We still fit the standard ensemble so Task 2 can extract the XGB model dict
        ensemble = StackingEnsemble()
        ensemble.fit(
            X_all, y_all,
            oof_preds_dict={'xgb': xgb_oof, 'lgbm': lgbm_oof, 'pinn': pinn_oof},
            base_models_dict={'xgb': xgb_final, 'lgbm': lgbm_final, 'pinn': pinn_final}
        )

        # Test predictions from base models
        test_preds = {
            'xgb':  xgb_final.predict(X_test),
            'lgbm': lgbm_final.predict(X_test),
            'pinn': pinn_final.predict(X_test),
        }

        # 🔥 ELITE UPGRADE: Hardcoded Top-Tier Weighted Blend
        print("Applying Elite Weighted Ensemble (0.6 XGB + 0.3 LGBM + 0.1 PINN)...")
        final_log_preds = (0.60 * test_preds['xgb']) + (0.30 * test_preds['lgbm']) + (0.10 * test_preds['pinn'])
        
        final_preds = inverse_transform_targets(final_log_preds)  # [n, 6]

        # Evaluate on validation set using the exact same ELITE weights
        val_preds_dict = {
            'xgb':  xgb_final.predict(X_val),
            'lgbm': lgbm_final.predict(X_val),
            'pinn': pinn_final.predict(X_val),
        }
        val_preds_log = (0.60 * val_preds_dict['xgb']) + (0.30 * val_preds_dict['lgbm']) + (0.10 * val_preds_dict['pinn'])
        
        TARGET_COLS = ['P80', 'R95', 'fines_frac', 'oversize_frac', 'R50_fines', 'R50_oversize']
        val_metrics = evaluate_predictions(y_val, val_preds_log, target_names=TARGET_COLS)
        
        # Dynamically log all 6 target metrics to MLflow
        metrics_to_log = {}
        for t in TARGET_COLS:
            metrics_to_log[f'val_rmse_{t}'] = val_metrics[t]['RMSE']
            metrics_to_log[f'val_r2_{t}'] = val_metrics[t]['R2']
            metrics_to_log[f'val_mae_{t}'] = val_metrics[t]['MAE']
        mlflow.log_metrics(metrics_to_log)

        # Generate Task 1 submission with all 6 columns
        submission = pd.DataFrame({'id': test_df['id']})
        for i, col in enumerate(TARGET_COLS):
            submission[col] = final_preds[:, i]
            
        sub_path = os.path.join(args.output_dir, 'task1_submission.csv')
        submission.to_csv(sub_path, index=False)
        print(f"\nTask 1 submission saved -> {sub_path}")
        mlflow.log_artifact(sub_path)

        # ── Task 2: Inverse Design ────────────────────────────
        if args.run_task2:
            print("\n[8/8] Running Task 2 inverse design...")

            from src.feature_engineering import engineer_physics_features, apply_log_transforms
            from src.models.pinn_model import ImpactPINN

            def feature_fn(df):
                df = engineer_physics_features(df)
                df = apply_log_transforms(df)
                from src.feature_engineering import get_feature_columns
                cols = get_feature_columns(df)
                return df[cols].values.astype(np.float32)

            predictor = EnsemblePredictor(ensemble, feature_fn, scaler)
            task2_path = os.path.join(args.output_dir, 'task2_scenarios.csv')
            df_task2 = run_inverse_design(predictor, output_path=task2_path)
            mlflow.log_artifact(task2_path)

        print("\n" + "=" * 60)
        print("✅ Pipeline complete! Elite configuration successfully applied.")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Boom Challenge Training Pipeline')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Directory containing train.csv, train_labels.csv, and test.csv')
    parser.add_argument('--output_dir', type=str, default='outputs/',
                        help='Directory for outputs, models, and submissions')
    parser.add_argument('--run_hpo', action='store_true',
                        help='Run Optuna hyperparameter optimization')
    parser.add_argument('--run_task2', action='store_true',
                        help='Run inverse design (Task 2)')
    parser.add_argument('--version', type=str, default='1',
                        help='Experiment version tag for MLflow')

    args = parser.parse_args()
    main(args)

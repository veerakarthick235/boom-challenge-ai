"""
boom_challenge/src/ensemble.py
================================
Stacking ensemble using NNLS meta learner.
Updated to support 6 targets and Mean Absolute Error (MAE) evaluation.
"""

import numpy as np
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingClassifier

SEED = 42
N_FOLDS = 5

TARGETS = ['P80', 'R95', 'fines_frac', 'oversize_frac', 'R50_fines', 'R50_oversize']
LOG_TARGETS = [f"log_{t}" for t in TARGETS]


# ─────────────────────────────────────────────
# OOF CROSS VALIDATION
# ─────────────────────────────────────────────

def generate_oof_predictions(
    model_class,
    model_kwargs,
    X,
    y,
    fit_kwargs=None,
    n_folds=N_FOLDS,
    model_name="model"
):

    oof_preds = np.zeros_like(y, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fit_kwargs = fit_kwargs or {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        print(f"  [{model_name}] Fold {fold+1}/{n_folds}")

        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        model = model_class(**model_kwargs)
        model.fit(X_tr, y_tr, X_vl, y_vl, **fit_kwargs)

        preds = model.predict(X_vl)
        oof_preds[val_idx] = preds

        # Print out MAE for each target during cross-validation
        for i, col in enumerate(LOG_TARGETS):
            mae = mean_absolute_error(y_vl[:, i], preds[:, i])
            print(f"     {col} MAE = {mae:.5f}")

    return oof_preds


# ─────────────────────────────────────────────
# STACKING ENSEMBLE
# ─────────────────────────────────────────────

class StackingEnsemble:

    def __init__(self, **kwargs):
        # Create a separate NNLS meta-learner for each of the 6 targets
        self.meta_learners = [
            LinearRegression(positive=True, fit_intercept=False) 
            for _ in range(len(TARGETS))
        ]
        self.base_models = {}

    # ─────────────────────────────

    def fit(self, X_train, y_train, oof_preds_dict, base_models_dict=None):
        
        if base_models_dict is not None:
            self.base_models = base_models_dict

        print("\n[Ensemble Weights]")
        for i, target_name in enumerate(TARGETS):
            # Extract the i-th target predictions from all base models
            meta_X = np.column_stack([v[:, i] for v in oof_preds_dict.values()])
            
            # Fit the NNLS model for this specific target
            self.meta_learners[i].fit(meta_X, y_train[:, i])
            
            print(f"  {target_name.ljust(15)}: {self.meta_learners[i].coef_}")

        return self

    # ─────────────────────────────

    def predict(self, test_preds):
        """
        Supports two inputs:
        1) dict of predictions (normal stacking)
        2) raw feature matrix (inverse design)
        """

        # case 1: stacking predictions
        if isinstance(test_preds, dict):
            source_dict = test_preds

        # case 2: raw features
        else:
            if not self.base_models:
                raise ValueError("Base models not loaded! Provide base_models_dict during fit().")
                
            source_dict = {}
            for name, model in self.base_models.items():
                source_dict[name] = model.predict(test_preds)

        final_preds = []
        for i in range(len(TARGETS)):
            meta_X = np.column_stack([v[:, i] for v in source_dict.values()])
            pred = self.meta_learners[i].predict(meta_X)
            final_preds.append(pred)

        return np.column_stack(final_preds)

    # ─────────────────────────────

    def save(self, path):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump({
            "meta_learners": self.meta_learners,
            "base_models": self.base_models  # Save base models for inference
        }, path)

    # ─────────────────────────────

    def load(self, path):

        data = joblib.load(path)

        self.meta_learners = data["meta_learners"]
        self.base_models = data.get("base_models", {})

        return self


# ─────────────────────────────────────────────
# ADVERSARIAL VALIDATION
# ─────────────────────────────────────────────

def adversarial_validation(X_train, X_test, feature_names=None):

    X_combined = np.vstack([X_train, X_test])

    y_combined = np.hstack([
        np.zeros(len(X_train)),
        np.ones(len(X_test))
    ])

    clf = GradientBoostingClassifier(
        n_estimators=100,
        random_state=SEED
    )

    auc_scores = cross_val_score(
        clf,
        X_combined,
        y_combined,
        cv=5,
        scoring="roc_auc"
    )

    mean_auc = auc_scores.mean()

    print(
        f"Adversarial AUC: {mean_auc:.4f} "
        f"{'⚠ Shift detected' if mean_auc > 0.55 else '✓ OK'}"
    )

    return {"mean_auc": mean_auc}


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_predictions(
    y_true,
    y_pred,
    target_names=TARGETS,
    log_space=True
):

    results = {}

    for i, name in enumerate(target_names):

        yt = y_true[:, i]
        yp = y_pred[:, i]

        if log_space:
            yt = np.expm1(yt)
            yp = np.expm1(yp)

        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)

        mape = np.mean(
            np.abs((yt - yp) / (np.abs(yt) + 1e-9))
        ) * 100

        print(f"{name.ljust(15)}: MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | MAPE={mape:.2f}%")

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        }

    return results
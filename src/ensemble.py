"""
boom_challenge/src/ensemble.py
================================
Stacking ensemble using NNLS meta learner.
"""

import numpy as np
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingClassifier

SEED = 42
N_FOLDS = 5


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

        for i, col in enumerate(["log_P80", "log_R95"]):
            rmse = np.sqrt(mean_squared_error(y_vl[:, i], preds[:, i]))
            print(f"     {col} RMSE = {rmse:.5f}")

    return oof_preds


# ─────────────────────────────────────────────
# STACKING ENSEMBLE
# ─────────────────────────────────────────────

class StackingEnsemble:

    def __init__(self, **kwargs):
        # **kwargs safely catches legacy arguments like `alpha=0.5` from train.py
        # LinearRegression with positive=True and fit_intercept=False creates NNLS
        self.meta_P80 = LinearRegression(positive=True, fit_intercept=False)
        self.meta_R95 = LinearRegression(positive=True, fit_intercept=False)

        # base models used for inverse design
        self.base_models = {}

    # ─────────────────────────────

    def fit(self, X_train, y_train, oof_preds_dict, base_models_dict=None):
        
        if base_models_dict is not None:
            self.base_models = base_models_dict

        meta_X_P80 = np.column_stack([v[:,0] for v in oof_preds_dict.values()])
        meta_X_R95 = np.column_stack([v[:,1] for v in oof_preds_dict.values()])

        # StandardScalers removed! NNLS requires raw prediction values.
        self.meta_P80.fit(meta_X_P80, y_train[:,0])
        self.meta_R95.fit(meta_X_R95, y_train[:,1])

        print("\nMeta learner weights (P80):", self.meta_P80.coef_)
        print("Meta learner weights (R95):", self.meta_R95.coef_)

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
            meta_X_P80 = np.column_stack([v[:,0] for v in test_preds.values()])
            meta_X_R95 = np.column_stack([v[:,1] for v in test_preds.values()])

        # case 2: raw features
        else:
            if not self.base_models:
                raise ValueError("Base models not loaded! Provide base_models_dict during fit().")
                
            preds = {}
            for name, model in self.base_models.items():
                preds[name] = model.predict(test_preds)

            meta_X_P80 = np.column_stack([v[:,0] for v in preds.values()])
            meta_X_R95 = np.column_stack([v[:,1] for v in preds.values()])

        # StandardScalers removed here as well
        final_P80 = self.meta_P80.predict(meta_X_P80)
        final_R95 = self.meta_R95.predict(meta_X_R95)

        return np.column_stack([final_P80, final_R95])

    # ─────────────────────────────

    def save(self, path):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump({
            "meta_P80": self.meta_P80,
            "meta_R95": self.meta_R95,
            "base_models": self.base_models  # Save base models for inference
        }, path)

    # ─────────────────────────────

    def load(self, path):

        data = joblib.load(path)

        self.meta_P80 = data["meta_P80"]
        self.meta_R95 = data["meta_R95"]
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
    target_names=("P80","R95"),
    log_space=True
):

    results = {}

    for i, name in enumerate(target_names):

        yt = y_true[:,i]
        yp = y_pred[:,i]

        if log_space:
            yt = np.expm1(yt)
            yp = np.expm1(yp)

        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)

        mape = np.mean(
            np.abs((yt - yp) / (np.abs(yt) + 1e-9))
        ) * 100

        print(f"{name}: RMSE={rmse:.4f} | R²={r2:.4f} | MAPE={mape:.2f}%")

        results[name] = {
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        }

    return results

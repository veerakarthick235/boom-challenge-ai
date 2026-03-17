"""
boom_challenge/src/models/gbm_models.py
=========================================
XGBoost and LightGBM trainers with Optuna hyperparameter optimization.
Supports multi-output regression (P80 and R95 trained separately).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import joblib
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42


# ─────────────────────────────────────────────────────────────
# XGBOOST MODEL
# ─────────────────────────────────────────────────────────────

class XGBModel:
    """XGBoost regressor wrapper with Optuna HPO."""

    DEFAULT_PARAMS = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 5,
        'gamma': 0.05,
        'random_state': SEED,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'n_jobs': -1,
        'early_stopping_rounds': 50,
    }

    def __init__(self, params: dict = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.models = {}  # {'P80': model, 'R95': model}
        self.feature_importances_ = {}

    def _train_one_target(self, X_train, y_train, X_val, y_val,
                           target_name: str) -> xgb.XGBRegressor:
        params = dict(self.params)
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return model

    def fit(self, X_train, y_train, X_val, y_val,
            target_cols: list = ['log_P80', 'log_R95']):
        """
        Train separate models for each target.

        y_train: DataFrame or array with columns target_cols
        """
        if isinstance(y_train, pd.DataFrame):
            y_train_arr = y_train[target_cols].values
            y_val_arr   = y_val[target_cols].values
        else:
            y_train_arr = y_train
            y_val_arr   = y_val

        for i, col in enumerate(target_cols):
            print(f"  Training XGBoost for {col}...")
            model = self._train_one_target(
                X_train, y_train_arr[:, i],
                X_val,   y_val_arr[:, i],
                target_name=col
            )
            self.models[col] = model
            self.feature_importances_[col] = model.feature_importances_

        return self

    def predict(self, X) -> np.ndarray:
        """Returns [n, n_targets] array of predictions."""
        preds = []
        for col, model in self.models.items():
            preds.append(model.predict(X))
        return np.column_stack(preds)

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        for name, model in self.models.items():
            model.save_model(os.path.join(dir_path, f'xgb_{name}.json'))

    def load(self, dir_path: str, target_cols: list = ['log_P80', 'log_R95']):
        for col in target_cols:
            path = os.path.join(dir_path, f'xgb_{col}.json')
            model = xgb.XGBRegressor()
            model.load_model(path)
            self.models[col] = model
        return self


def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                 n_trials: int = 100, cv: int = 5) -> dict:
    """
    Optuna hyperparameter search for XGBoost.
    Returns best parameter dict.
    """
    def objective(trial):
        params = {
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('lr', 0.005, 0.3, log=True),
            'n_estimators':     trial.suggest_int('n_estimators', 300, 3000),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('cbt', 0.4, 1.0),
            'reg_alpha':        trial.suggest_float('alpha', 1e-5, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('lambda', 1e-5, 10.0, log=True),
            'min_child_weight': trial.suggest_int('mcw', 1, 20),
            'gamma':            trial.suggest_float('gamma', 0.0, 1.0),
            'random_state':     SEED,
            'objective':        'reg:squarederror',
            'tree_method':      'hist',
            'n_jobs':           -1,
        }
        model = xgb.XGBRegressor(**params)
        kf = KFold(n_splits=cv, shuffle=True, random_state=SEED)
        scores = cross_val_score(model, X_train, y_train, cv=kf,
                                 scoring='neg_root_mean_squared_error',
                                 n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"XGBoost best RMSE: {study.best_value:.5f}")
    return study.best_params


# ─────────────────────────────────────────────────────────────
# LIGHTGBM MODEL
# ─────────────────────────────────────────────────────────────

class LGBMModel:
    """LightGBM regressor wrapper with Optuna HPO."""

    DEFAULT_PARAMS = {
        'n_estimators': 1500,
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 20,
        'max_depth': -1,
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1,
    }

    def __init__(self, params: dict = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.models = {}

    def fit(self, X_train, y_train, X_val, y_val,
            target_cols: list = ['log_P80', 'log_R95']):
        if isinstance(y_train, pd.DataFrame):
            y_train_arr = y_train[target_cols].values
            y_val_arr   = y_val[target_cols].values
        else:
            y_train_arr = y_train
            y_val_arr   = y_val

        for i, col in enumerate(target_cols):
            print(f"  Training LightGBM for {col}...")
            callbacks = [lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)]
            model = lgb.LGBMRegressor(**self.params)
            model.fit(
                X_train, y_train_arr[:, i],
                eval_set=[(X_val, y_val_arr[:, i])],
                callbacks=callbacks,
            )
            self.models[col] = model
        return self

    def predict(self, X) -> np.ndarray:
        preds = [m.predict(X) for m in self.models.values()]
        return np.column_stack(preds)

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(dir_path, f'lgbm_{name}.pkl'))

    def load(self, dir_path: str, target_cols: list = ['log_P80', 'log_R95']):
        for col in target_cols:
            self.models[col] = joblib.load(
                os.path.join(dir_path, f'lgbm_{col}.pkl'))
        return self


def tune_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                  n_trials: int = 100, cv: int = 5) -> dict:
    """Optuna HPO for LightGBM."""
    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 300, 3000),
            'num_leaves':       trial.suggest_int('num_leaves', 20, 300),
            'learning_rate':    trial.suggest_float('lr', 0.005, 0.3, log=True),
            'feature_fraction': trial.suggest_float('ff', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bf', 0.4, 1.0),
            'bagging_freq':     trial.suggest_int('bfq', 1, 10),
            'reg_alpha':        trial.suggest_float('alpha', 1e-5, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('lambda', 1e-5, 10.0, log=True),
            'min_child_samples': trial.suggest_int('mcs', 5, 100),
            'random_state': SEED, 'n_jobs': -1, 'verbose': -1,
        }
        model = lgb.LGBMRegressor(**params)
        kf = KFold(n_splits=cv, shuffle=True, random_state=SEED)
        scores = cross_val_score(model, X_train, y_train, cv=kf,
                                 scoring='neg_root_mean_squared_error',
                                 n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"LightGBM best RMSE: {study.best_value:.5f}")
    return study.best_params

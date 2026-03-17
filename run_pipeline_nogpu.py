"""
boom_challenge/run_pipeline_nogpu.py
=====================================
CPU-only pipeline: XGBoost + LightGBM + RandomForest + Stacking
(PyTorch-free — works on any Python 3.11 Windows machine)

Usage:
    python run_pipeline_nogpu.py
    python run_pipeline_nogpu.py --run_hpo
    python run_pipeline_nogpu.py --run_hpo --run_task2
"""

import argparse, os, random, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.optimize import minimize
import joblib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
random.seed(SEED); np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════
# 1. PHYSICS FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); eps = 1e-12

    rho_t_si = df['rho_t'] * 1000
    rho_i_si = df['rho_i'] * 1000

    df['KE']          = 0.5 * df['m_i'] * df['v_i']**2
    df['momentum']    = df['m_i'] * df['v_i']
    df['rho_ratio']   = rho_i_si / (rho_t_si + eps)
    df['E_spec']      = df['KE'] / (rho_t_si * df['d_i']**3 + eps)
    df['pi_v']        = df['v_i'] / (np.sqrt(df['Y'] / (rho_t_si + eps)) + eps)
    df['pi_2']        = (rho_t_si * df['g'] * df['d_i']) / (df['Y'] + eps)
    df['pi_grav']     = (df['g'] * df['d_i']) / (df['v_i']**2 + eps)
    df['mu_crater']   = np.clip(
        (df['KE'] / (df['Y'] * df['d_i']**3 + eps))**(1/3) * df['rho_ratio']**(1/3),
        1e-6, 1e6)
    df['impedance']   = rho_i_si * df['v_i']
    df['KE_area']     = df['KE'] / (np.pi/4 * df['d_i']**2 + eps)
    df['mass_str']    = df['m_i'] / (df['Y'] * df['d_i']**2 + eps)

    df['sin_t']  = np.sin(np.radians(df['theta']))
    df['cos_t']  = np.cos(np.radians(df['theta']))
    df['sin2_t'] = np.sin(2 * np.radians(df['theta']))

    # Normalized pi-groups
    df['npi_v']  = (np.log10(np.clip(df['pi_v'],  1e-4, 50))   + 2.5) / 4.0
    df['npi_2']  = (np.log10(np.clip(df['pi_2'],  1e-8, 20))   + 7.0) / 8.0
    df['nmu_c']  = (np.log10(np.clip(df['mu_crater'], 1e-3, 5e3)) + 2.0) / 5.0
    df['nY']     = (np.log10(df['Y'])  - 5.0) / 5.0
    df['ng']     = (np.log10(df['g'] + 0.01) + 1.3) / 2.6

    # Interactions
    df['v_m']    = np.log1p(df['v_i']) * np.log1p(df['m_i'])
    df['KE_sin'] = np.log1p(df['KE']) * df['sin_t']

    # Log transforms
    for c in ['v_i','m_i','Y','g','d_i','KE','momentum','E_spec',
              'pi_v','pi_2','mu_crater','KE_area','mass_str','impedance']:
        df[f'log_{c}'] = np.log1p(np.abs(df[c]))

    return df


FEAT_DROP = {'P80','R95','log_P80','log_R95'}

def get_feat_cols(df):
    return [c for c in df.columns
            if c not in FEAT_DROP and df[c].dtype != object]


def preprocess(train_df, test_df):
    tr = engineer_features(train_df)
    te = engineer_features(test_df)
    tr['log_P80'] = np.log1p(tr['P80'])
    tr['log_R95'] = np.log1p(tr['R95'])

    feat_cols = get_feat_cols(tr)

    X_all = np.nan_to_num(tr[feat_cols].values.astype(np.float32),
                           nan=0.0, posinf=1e6, neginf=-1e6)
    y_all = tr[['log_P80','log_R95']].values.astype(np.float32)

    scaler = RobustScaler().fit(X_all)
    X_all  = scaler.transform(X_all)

    X_te = np.nan_to_num(te[feat_cols].values.astype(np.float32),
                          nan=0.0, posinf=1e6, neginf=-1e6)
    X_te = scaler.transform(X_te)
    print(f"  Features: {len(feat_cols)}  | Train: {X_all.shape[0]} | Test: {X_te.shape[0]}")
    return X_all, y_all, X_te, feat_cols, scaler


# ═══════════════════════════════════════════════════════════════
# 2. MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

def train_xgb(Xtr, ytr, Xvl, yvl, params=None):
    base = dict(n_estimators=800, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=SEED,
                objective='reg:squarederror', tree_method='hist',
                n_jobs=-1, early_stopping_rounds=50)
    if params: base.update(params)
    mdls = {}
    for i, t in enumerate(['P80','R95']):
        m = xgb.XGBRegressor(**base)
        m.fit(Xtr, ytr[:,i], eval_set=[(Xvl, yvl[:,i])], verbose=False)
        mdls[t] = m
    return mdls


def train_lgbm(Xtr, ytr, Xvl, yvl, params=None):
    base = dict(n_estimators=1000, num_leaves=63, learning_rate=0.03,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
                random_state=SEED, n_jobs=-1, verbose=-1)
    if params: base.update(params)
    cbs = [lgb.early_stopping(50,verbose=False), lgb.log_evaluation(period=-1)]
    mdls = {}
    for i, t in enumerate(['P80','R95']):
        m = lgb.LGBMRegressor(**base)
        m.fit(Xtr, ytr[:,i], eval_set=[(Xvl, yvl[:,i])], callbacks=cbs)
        mdls[t] = m
    return mdls


def train_rf(Xtr, ytr):
    mdls = {}
    for i, t in enumerate(['P80','R95']):
        m = RandomForestRegressor(n_estimators=300, max_depth=12,
                                   min_samples_leaf=5, random_state=SEED,
                                   n_jobs=-1)
        m.fit(Xtr, ytr[:,i])
        mdls[t] = m
    return mdls


def predict_dict(mdls, X):
    return np.column_stack([mdls['P80'].predict(X),
                             mdls['R95'].predict(X)])


# ═══════════════════════════════════════════════════════════════
# 3. OPTUNA HPO
# ═══════════════════════════════════════════════════════════════

def optuna_xgb(X, y, n_trials=25):
    def obj(trial):
        p = dict(max_depth=trial.suggest_int('md',3,8),
                 learning_rate=trial.suggest_float('lr',0.01,0.3,log=True),
                 n_estimators=trial.suggest_int('ne',200,1000),
                 subsample=trial.suggest_float('ss',0.6,1.0),
                 colsample_bytree=trial.suggest_float('cbt',0.5,1.0),
                 reg_alpha=trial.suggest_float('ra',1e-4,5.0,log=True),
                 reg_lambda=trial.suggest_float('rl',1e-4,5.0,log=True),
                 random_state=SEED, objective='reg:squarederror',
                 tree_method='hist', n_jobs=-1)
        s = cross_val_score(xgb.XGBRegressor(**p), X, y, cv=3,
                            scoring='neg_root_mean_squared_error',n_jobs=-1)
        return -s.mean()
    st = optuna.create_study(direction='minimize',
                              sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"  XGB best log-RMSE: {st.best_value:.5f}")
    return st.best_params


def optuna_lgbm(X, y, n_trials=25):
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int('ne',200,1200),
                 num_leaves=trial.suggest_int('nl',20,150),
                 learning_rate=trial.suggest_float('lr',0.01,0.3,log=True),
                 feature_fraction=trial.suggest_float('ff',0.5,1.0),
                 bagging_fraction=trial.suggest_float('bf',0.5,1.0),
                 bagging_freq=trial.suggest_int('bfq',1,8),
                 reg_alpha=trial.suggest_float('ra',1e-4,5.0,log=True),
                 reg_lambda=trial.suggest_float('rl',1e-4,5.0,log=True),
                 random_state=SEED, n_jobs=-1, verbose=-1)
        s = cross_val_score(lgb.LGBMRegressor(**p), X, y, cv=3,
                            scoring='neg_root_mean_squared_error',n_jobs=-1)
        return -s.mean()
    st = optuna.create_study(direction='minimize',
                              sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"  LGBM best log-RMSE: {st.best_value:.5f}")
    return st.best_params


# ═══════════════════════════════════════════════════════════════
# 4. OOF STACKING
# ═══════════════════════════════════════════════════════════════

def generate_oof(X, y, n_folds=5, xgb_params=None, lgbm_params=None):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_xgb = np.zeros_like(y); oof_lgbm = np.zeros_like(y)
    oof_rf  = np.zeros_like(y)

    for fold, (tri, vli) in enumerate(kf.split(X)):
        Xtr, Xvl = X[tri], X[vli]
        ytr, yvl = y[tri], y[vli]

        xm = train_xgb(Xtr, ytr, Xvl, yvl, xgb_params)
        lm = train_lgbm(Xtr, ytr, Xvl, yvl, lgbm_params)
        rm = train_rf(Xtr, ytr)

        oof_xgb[vli]  = predict_dict(xm, Xvl)
        oof_lgbm[vli] = predict_dict(lm, Xvl)
        oof_rf[vli]   = predict_dict(rm, Xvl)

        # Per-fold metrics
        for i, t in enumerate(['P80','R95']):
            rxgb  = np.sqrt(mean_squared_error(yvl[:,i], oof_xgb[vli,i]))
            rlgbm = np.sqrt(mean_squared_error(yvl[:,i], oof_lgbm[vli,i]))
            rrf   = np.sqrt(mean_squared_error(yvl[:,i], oof_rf[vli,i]))
            print(f"  Fold {fold+1} {t}: XGB={rxgb:.4f} LGBM={rlgbm:.4f} RF={rrf:.4f}")

    return oof_xgb, oof_lgbm, oof_rf


def train_meta(oof_xgb, oof_lgbm, oof_rf, y_all):
    meta = {}
    for i, t in enumerate(['P80','R95']):
        Xm = np.column_stack([oof_xgb[:,i], oof_lgbm[:,i], oof_rf[:,i]])
        m  = Ridge(alpha=0.5).fit(Xm, y_all[:,i])
        meta[t] = m
        print(f"  Meta [{t}]: "
              f"XGB={m.coef_[0]:.3f} LGBM={m.coef_[1]:.3f} RF={m.coef_[2]:.3f}")
    return meta


def ensemble_predict(meta, xpred, lpred, rpred):
    out = np.zeros((xpred.shape[0], 2))
    for i, t in enumerate(['P80','R95']):
        Xm = np.column_stack([xpred[:,i], lpred[:,i], rpred[:,i]])
        out[:,i] = meta[t].predict(Xm)
    return out


# ═══════════════════════════════════════════════════════════════
# 5. EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate(y_true_log, y_pred_log, label='Validation'):
    print(f"\n{'='*50}\n📊 {label} Results\n{'='*50}")
    metrics = {}
    for i, t in enumerate(['P80','R95']):
        yt = np.expm1(y_true_log[:,i]); yp = np.expm1(y_pred_log[:,i])
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2   = r2_score(yt, yp)
        mape = np.mean(np.abs((yt-yp)/(np.abs(yt)+1e-9)))*100
        print(f"  {t}: RMSE={rmse:.3f}  R²={r2:.4f}  MAPE={mape:.2f}%")
        metrics[t] = {'RMSE':rmse,'R2':r2,'MAPE':mape}
    return metrics


# ═══════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot_all(train_df, y_val_log, val_pred_log, feat_cols,
             xgb_models, output_dir):
    os.makedirs(f'{output_dir}/eda',   exist_ok=True)
    os.makedirs(f'{output_dir}/plots', exist_ok=True)

    # Target distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    train_df['P80'].hist(bins=50, ax=axes[0], color='steelblue', edgecolor='white')
    axes[0].set_title('P80 Distribution (Fragment Size)'); axes[0].set_xlabel('P80 (m)')
    train_df['R95'].hist(bins=50, ax=axes[1], color='coral', edgecolor='white')
    axes[1].set_title('R95 Distribution (Ejecta Radius)'); axes[1].set_xlabel('R95 (m)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda/target_distributions.png', dpi=150)
    plt.close()
    print(f"  Saved: target_distributions.png")

    # Correlation heatmap
    tr_eng = engineer_features(train_df)
    num_cols = [c for c in tr_eng.columns if tr_eng[c].dtype != object
                and c not in {'log_P80','log_R95'}][:20]
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(tr_eng[num_cols].corr(), annot=False, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda/correlation_heatmap.png', dpi=150)
    plt.close()
    print(f"  Saved: correlation_heatmap.png")

    # Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (t, c) in enumerate([('P80','steelblue'), ('R95','coral')]):
        yt = np.expm1(y_val_log[:,i])
        yp = np.expm1(val_pred_log[:,i])
        axes[i].scatter(yt, yp, s=8, alpha=0.5, color=c)
        lim = [min(yt.min(),yp.min())*0.95, max(yt.max(),yp.max())*1.05]
        axes[i].plot(lim, lim, 'k--', lw=1.5, label='Perfect fit')
        axes[i].set_xlabel(f'Actual {t}'); axes[i].set_ylabel(f'Predicted {t}')
        r2 = r2_score(yt, yp)
        axes[i].set_title(f'{t}: Actual vs Predicted  (R²={r2:.4f})')
        axes[i].legend(); axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/actual_vs_predicted.png', dpi=150)
    plt.close()
    print(f"  Saved: actual_vs_predicted.png")

    # Feature importance from XGBoost
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, t in enumerate(['P80', 'R95']):
        imp = xgb_models[t].feature_importances_
        top_idx = np.argsort(imp)[::-1][:20]
        top_imp = imp[top_idx]
        top_names = [feat_cols[j] for j in top_idx]
        axes[i].barh(range(len(top_imp)), top_imp[::-1], color='steelblue' if i==0 else 'coral')
        axes[i].set_yticks(range(len(top_imp)))
        axes[i].set_yticklabels(top_names[::-1], fontsize=8)
        axes[i].set_title(f'XGBoost Feature Importance — {t}')
        axes[i].set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/feature_importance.png', dpi=150)
    plt.close()
    print(f"  Saved: feature_importance.png")


# ═══════════════════════════════════════════════════════════════
# 7. INVERSE DESIGN (TASK 2)
# ═══════════════════════════════════════════════════════════════

PARAM_NAMES = ['v_i','m_i','rho_i','rho_t','Y','g','d_i','theta']
LB = np.array([500,   1e6,  1.5, 1.5, 1e5, 0.05,  10, 10], dtype=float)
UB = np.array([25000, 1e12, 8.0, 3.5, 1e10, 20.0, 8000, 80], dtype=float)


def make_predictor(xgb_final, lgbm_final, rf_final, meta, feat_cols, scaler):
    def predict_single(x):
        df = pd.DataFrame([x], columns=PARAM_NAMES)
        dfe = engineer_features(df)
        for c in feat_cols:
            if c not in dfe.columns: dfe[c] = 0.0
        X = np.nan_to_num(dfe[feat_cols].values.astype(np.float32),
                           nan=0.0, posinf=1e6, neginf=-1e6)
        X = scaler.transform(X)
        xp = predict_dict(xgb_final, X)
        lp = predict_dict(lgbm_final, X)
        rp = predict_dict(rf_final,   X)
        lp_ens = ensemble_predict(meta, xp, lp, rp)
        return float(np.expm1(lp_ens[0,0])), float(np.expm1(lp_ens[0,1]))
    return predict_single


def run_inverse_design(predict_fn, n_starts=150, output_dir='outputs'):
    print("\nSearching for feasible impact configurations...")

    def obj(x):
        P80, R95 = predict_fn(x)
        KE = 0.5 * x[1] * x[0]**2
        return KE/1e17 + 3.0*R95

    constraints = [
        {'type':'ineq','fun': lambda x, f=predict_fn: f(x)[0] - 96},
        {'type':'ineq','fun': lambda x, f=predict_fn: 101 - f(x)[0]},
        {'type':'ineq','fun': lambda x, f=predict_fn: 175 - f(x)[1]},
    ]
    bounds = list(zip(LB, UB))
    feasible = []

    for s in range(n_starts):
        np.random.seed(s)
        x0 = LB + np.random.rand(len(LB)) * (UB - LB)
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter':300,'ftol':1e-7})
            P80, R95 = predict_fn(res.x)
            if 94 <= P80 <= 103 and R95 <= 178:
                feasible.append({'x':res.x,'P80':P80,'R95':R95,
                                  'KE':0.5*res.x[1]*res.x[0]**2})
        except Exception:
            pass
        if s % 30 == 0:
            print(f"  Start {s}/{n_starts} — feasible so far: {len(feasible)}")
        if len(feasible) >= 60:
            break

    print(f"  Total feasible candidates: {len(feasible)}")

    # fallback: random feasible sampling
    if len(feasible) < 20:
        print("  Running random feasible sampling as fallback...")
        for s in range(2000):
            np.random.seed(s + 9999)
            x0 = LB + np.random.rand(len(LB)) * (UB - LB)
            P80, R95 = predict_fn(x0)
            if 90 <= P80 <= 105 and R95 <= 185:
                feasible.append({'x':x0,'P80':P80,'R95':R95,
                                  'KE':0.5*x0[1]*x0[0]**2})
            if len(feasible) >= 40: break

    # Select 20 diverse solutions
    n_sel = min(20, len(feasible))
    feasible.sort(key=lambda s: s['KE']/1e17 + s['R95'])
    stride  = max(1, len(feasible) // n_sel)
    sel = feasible[::stride][:n_sel]
    # pad if needed
    if len(sel) < n_sel:
        sel += feasible[-(n_sel-len(sel)):]
    sel = sel[:n_sel]

    rows = []
    for i, sol in enumerate(sel):
        row = {PARAM_NAMES[j]: sol['x'][j] for j in range(len(PARAM_NAMES))}
        row.update({'P80_pred':round(sol['P80'],3),'R95_pred':round(sol['R95'],3),
                    'KE_J':sol['KE'],'scenario_id':i+1})
        rows.append(row)

    df_out = pd.DataFrame(rows)
    path = f'{output_dir}/task2_scenarios.csv'
    df_out.to_csv(path, index=False)

    feasible_mask = (df_out['P80_pred']>=96) & (df_out['P80_pred']<=101) & \
                    (df_out['R95_pred']<=175)
    print(f"\n✅ Task 2 — {len(df_out)} scenarios saved → {path}")
    print(f"   Scenarios satisfying strict constraints: {feasible_mask.sum()}/20")
    print(f"   P80 range: [{df_out['P80_pred'].min():.2f}, {df_out['P80_pred'].max():.2f}]")
    print(f"   R95 range: [{df_out['R95_pred'].min():.2f}, {df_out['R95_pred'].max():.2f}]")
    print(f"   KE  range: [{df_out['KE_J'].min():.2e}, {df_out['KE_J'].max():.2e}] J")

    # Pareto plot
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df_out['KE_J']/1e17, df_out['R95_pred'],
                    c=df_out['P80_pred'], cmap='plasma', s=180, edgecolors='k', zorder=5)
    plt.colorbar(sc, ax=ax, label='P80 (m)')
    ax.axhline(175, color='red', ls='--', lw=2, label='R95 ≤ 175 limit')
    ax.axvline(0, color='gray', ls='-', lw=0.5)
    for _, row in df_out.iterrows():
        ax.annotate(str(int(row['scenario_id'])),
                    (row['KE_J']/1e17, row['R95_pred']),
                    fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('Kinetic Energy (×10¹⁷ J)'); ax.set_ylabel('R95 (m)')
    ax.set_title('Task 2: 20 Optimal Impact Configurations\n(Color = P80 fragment size)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/task2_pareto.png', dpi=150)
    plt.close()
    print(f"   Saved: task2_pareto.png")

    return df_out


# ═══════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════

def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/models', exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────
    print("\n" + "="*55)
    print("[1/7] Loading data...")
    print("="*55)
    train_df = pd.read_csv(f'{args.data_dir}/train.csv')
    test_df  = pd.read_csv(f'{args.data_dir}/test.csv')
    print(f"  Train: {train_df.shape}  | Test: {test_df.shape}")
    print(f"  P80: mean={train_df['P80'].mean():.2f} std={train_df['P80'].std():.2f}")
    print(f"  R95: mean={train_df['R95'].mean():.2f} std={train_df['R95'].std():.2f}")

    # ── 2. Features ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("[2/7] Physics feature engineering...")
    print("="*55)
    X_all, y_all, X_test, feat_cols, scaler = preprocess(train_df, test_df)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.15, random_state=SEED)

    # ── 3. HPO ───────────────────────────────────────────────────
    xgb_params, lgbm_params = {}, {}
    if args.run_hpo:
        print("\n" + "="*55)
        print("[3/7] Optuna HPO (25 trials each)...")
        print("="*55)
        print("  Tuning XGBoost...")
        xgb_params  = optuna_xgb(X_tr, y_tr[:,0], n_trials=25)
        print("  Tuning LightGBM...")
        lgbm_params = optuna_lgbm(X_tr, y_tr[:,0], n_trials=25)
    else:
        print("\n[3/7] Using default hyperparameters (add --run_hpo for Optuna tuning)")

    # ── 4. OOF ──────────────────────────────────────────────────
    print("\n" + "="*55)
    print("[4/7] 5-Fold OOF cross-validation (XGB + LGBM + RF)...")
    print("="*55)
    oof_xgb, oof_lgbm, oof_rf = generate_oof(
        X_all, y_all, n_folds=5, xgb_params=xgb_params, lgbm_params=lgbm_params)

    print("\n  Overall OOF (log-space → original space):")
    for i, t in enumerate(['P80','R95']):
        for nm, oof in [('XGB',oof_xgb),('LGBM',oof_lgbm),('RF',oof_rf)]:
            yt = np.expm1(y_all[:,i]); yp = np.expm1(oof[:,i])
            print(f"  {nm} {t}: RMSE={np.sqrt(mean_squared_error(yt,yp)):.3f}"
                  f"  R²={r2_score(yt,yp):.4f}")

    print("\n  Training meta-learner (Ridge stacking)...")
    meta = train_meta(oof_xgb, oof_lgbm, oof_rf, y_all)

    # ── 5. Final Models ──────────────────────────────────────────
    print("\n" + "="*55)
    print("[5/7] Training final models...")
    print("="*55)
    xgb_final  = train_xgb( X_tr, y_tr, X_val, y_val, xgb_params)
    lgbm_final = train_lgbm(X_tr, y_tr, X_val, y_val, lgbm_params)
    rf_final   = train_rf(  X_tr, y_tr)

    for t in ['P80','R95']:
        joblib.dump(xgb_final[t],  f'{output_dir}/models/xgb_{t}.pkl')
        joblib.dump(lgbm_final[t], f'{output_dir}/models/lgbm_{t}.pkl')
        joblib.dump(rf_final[t],   f'{output_dir}/models/rf_{t}.pkl')
    joblib.dump(scaler, f'{output_dir}/models/scaler.pkl')
    joblib.dump(meta,   f'{output_dir}/models/meta.pkl')
    joblib.dump(feat_cols, f'{output_dir}/models/feature_cols.pkl')
    print(f"  Models saved → {output_dir}/models/")

    # ── 6. Evaluation + Submission ───────────────────────────────
    print("\n" + "="*55)
    print("[6/7] Evaluating & generating Task 1 submission...")
    print("="*55)
    val_xgb  = predict_dict(xgb_final,  X_val)
    val_lgbm = predict_dict(lgbm_final, X_val)
    val_rf   = predict_dict(rf_final,   X_val)
    val_ens  = ensemble_predict(meta, val_xgb, val_lgbm, val_rf)
    metrics  = evaluate(y_val, val_ens, label='Validation (Ensemble)')

    plot_all(train_df, y_val, val_ens, feat_cols, xgb_final, output_dir)

    te_xgb  = predict_dict(xgb_final,  X_test)
    te_lgbm = predict_dict(lgbm_final, X_test)
    te_rf   = predict_dict(rf_final,   X_test)
    te_ens  = ensemble_predict(meta, te_xgb, te_lgbm, te_rf)
    te_orig = np.expm1(te_ens)

    sub = pd.DataFrame({'id': range(len(te_orig)),
                        'P80': te_orig[:,0], 'R95': te_orig[:,1]})
    sub_path = f'{output_dir}/task1_submission.csv'
    sub.to_csv(sub_path, index=False)
    print(f"\n  Task 1 submission → {sub_path}")
    print(f"  Predicted P80: mean={sub['P80'].mean():.2f}  std={sub['P80'].std():.2f}")
    print(f"  Predicted R95: mean={sub['R95'].mean():.2f}  std={sub['R95'].std():.2f}")

    # ── 7. Task 2 ─────────────────────────────────────────────────
    if args.run_task2:
        print("\n" + "="*55)
        print("[7/7] Task 2 — Inverse Design Optimization...")
        print("="*55)
        predict_fn = make_predictor(xgb_final, lgbm_final, rf_final,
                                     meta, feat_cols, scaler)
        run_inverse_design(predict_fn, n_starts=args.n_starts,
                            output_dir=output_dir)
    else:
        print("\n[7/7] Skipping Task 2 (add --run_task2 to enable)")

    print("\n" + "="*55)
    print("✅ PIPELINE COMPLETE!")
    print("="*55)
    print(f"\n  📂 All outputs → {output_dir}/")
    print(f"  📄 Task 1:  {output_dir}/task1_submission.csv")
    if args.run_task2:
        print(f"  📄 Task 2:  {output_dir}/task2_scenarios.csv")
    print(f"  📊 Plots:   {output_dir}/plots/  and  {output_dir}/eda/")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='data')
    p.add_argument('--output_dir', default='outputs')
    p.add_argument('--run_hpo',    action='store_true')
    p.add_argument('--run_task2',  action='store_true')
    p.add_argument('--n_starts',   type=int, default=150)
    main(p.parse_args())

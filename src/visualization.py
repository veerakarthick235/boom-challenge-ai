"""
boom_challenge/src/visualization.py
=====================================
Visualization utilities for EDA, model diagnostics, SHAP, 
physics relationships, and inverse design Pareto plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import os

plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d2e',
    'axes.edgecolor': '#3a3f5c',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'text.color': '#e0e0e0',
    'grid.color': '#2a2d3e',
    'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans'
})

PALETTE = ['#4ecdc4', '#ff6b6b', '#ffd93d', '#a8e6cf', '#c7a9ff']


def plot_target_distributions(df: pd.DataFrame, output_dir: str):
    """Plot P80 and R95 distributions — raw and log space."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Ejecta Target Distributions', fontsize=16, fontweight='bold')

    for col, color, row in [('P80', PALETTE[0], 0), ('R95', PALETTE[1], 1)]:
        axes[row, 0].hist(df[col], bins=60, color=color, edgecolor='black',
                          alpha=0.85)
        axes[row, 0].set_title(f'{col} — Raw Space')
        axes[row, 0].set_xlabel(col)
        axes[row, 0].set_ylabel('Count')
        axes[row, 0].grid(True)

        axes[row, 1].hist(np.log1p(df[col]), bins=60, color=color,
                          edgecolor='black', alpha=0.85)
        axes[row, 1].set_title(f'log(1 + {col}) — Log Space')
        axes[row, 1].set_xlabel(f'log1p({col})')
        axes[row, 1].grid(True)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'target_distributions.png'), dpi=150)
    plt.close()


def plot_physics_relationships(df: pd.DataFrame, output_dir: str):
    """Plot log-log scatter plots of physics features vs targets."""
    physics_feats = ['KE', 'pi_v', 'mu_crater', 'E_spec', 'density_ratio']
    targets = ['P80', 'R95']

    fig, axes = plt.subplots(len(physics_feats), 2,
                              figsize=(14, 4 * len(physics_feats)))
    fig.suptitle('Physics Feature vs Target (Log-Log Space)', fontsize=15,
                  fontweight='bold')

    for i, feat in enumerate(physics_feats):
        if feat not in df.columns:
            continue
        for j, target in enumerate(targets):
            x = np.log1p(np.abs(df[feat]))
            y = np.log1p(df[target])
            axes[i, j].scatter(x, y, s=4, alpha=0.4, color=PALETTE[j % len(PALETTE)])
            # Fit trend line
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 10:
                coeffs = np.polyfit(x[mask], y[mask], 1)
                xline = np.linspace(x[mask].min(), x[mask].max(), 100)
                axes[i, j].plot(xline, np.polyval(coeffs, xline),
                                 color='white', lw=2, ls='--',
                                 label=f'slope={coeffs[0]:.2f}')
                axes[i, j].legend(fontsize=8)
            axes[i, j].set_xlabel(f'log1p({feat})')
            axes[i, j].set_ylabel(f'log1p({target})')
            axes[i, j].set_title(f'{feat} vs {target}')
            axes[i, j].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'physics_relationships.png'), dpi=150)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                    target_names: list, output_dir: str):
    """Residual diagnostic plots for each target."""
    n = len(target_names)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, name in enumerate(target_names):
        yt = np.expm1(y_true[:, i])
        yp = np.expm1(y_pred[:, i])
        res = yt - yp

        # Predicted vs actual
        ax = axes[i, 0]
        ax.scatter(yt, yp, s=5, alpha=0.5, color=PALETTE[i % len(PALETTE)])
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, 'w--', lw=2, label='Perfect fit')
        ax.set_xlabel(f'Actual {name}'); ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}: Predicted vs Actual')
        ax.legend(); ax.grid(True)

        # Residuals vs predicted
        ax = axes[i, 1]
        ax.scatter(yp, res, s=5, alpha=0.5, color=PALETTE[(i + 2) % len(PALETTE)])
        ax.axhline(0, color='white', lw=2, ls='--')
        ax.set_xlabel(f'Predicted {name}'); ax.set_ylabel('Residual')
        ax.set_title(f'{name}: Residual Plot')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_plots.png'), dpi=150)
    plt.close()


def plot_shap_summary(model, X: np.ndarray, feature_names: list,
                       model_name: str, output_dir: str):
    """SHAP summary plot for a tree-based model."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:500])  # limit for speed

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X[:500], feature_names=feature_names,
                           plot_type='bar', show=False, max_display=20,
                           color='#4ecdc4')
        plt.title(f'SHAP Feature Importance — {model_name}', fontsize=14,
                   color='white', pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_{model_name.lower()}.png'),
                     dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"SHAP plot failed: {e}")


def plot_training_curves(history: dict, output_dir: str):
    """Plot PINN train/val loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], color=PALETTE[0], lw=2, label='Train Loss')
    ax.plot(epochs, history['val_loss'],   color=PALETTE[1], lw=2, label='Val Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss (log-space)')
    ax.set_title('PINN Training Curves', fontsize=14)
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pinn_training_curves.png'), dpi=150)
    plt.close()


def plot_inverse_design_pareto(df_solutions: pd.DataFrame, output_dir: str):
    """
    Visualize the 20 inverse design solutions on a 
    KE vs R95 scatter with P80 as colormap.
    """
    fig = plt.figure(figsize=(15, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1])

    # Pareto scatter
    ax1 = fig.add_subplot(gs[0])
    sc = ax1.scatter(df_solutions['KE_J'] / 1e15, df_solutions['R95'],
                      c=df_solutions['P80'], cmap='plasma',
                      s=200, edgecolors='white', linewidths=1.5, zorder=5)
    plt.colorbar(sc, ax=ax1, label='P80 (m)')

    # Constraint lines
    ax1.axhline(175, color='#ff6b6b', lw=2, ls='--', label='R95 ≤ 175 limit')
    for i, row in df_solutions.iterrows():
        ax1.annotate(str(int(row['scenario_id'])),
                      (row['KE_J'] / 1e15, row['R95']),
                      fontsize=8, color='white', ha='center', va='bottom')

    ax1.set_xlabel('Kinetic Energy (×10¹⁵ J)', fontsize=12)
    ax1.set_ylabel('R95 (m)', fontsize=12)
    ax1.set_title('Task 2: 20 Optimal Impact Configurations', fontsize=13,
                   fontweight='bold')
    ax1.legend(); ax1.grid(True)

    # P80 vs R95 scatter
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(df_solutions['P80'], df_solutions['R95'],
                 c=PALETTE[2], s=150, edgecolors='white', lw=1.5)
    ax2.axvline(96,  color='#ff6b6b', lw=1.5, ls='--', label='P80=96')
    ax2.axvline(101, color='#ff6b6b', lw=1.5, ls='--', label='P80=101')
    ax2.axhline(175, color='#4ecdc4', lw=1.5, ls='--', label='R95=175')
    ax2.fill_betweenx([0, 175], 96, 101, alpha=0.12, color='#ffd93d',
                       label='Feasible region')
    ax2.set_xlabel('P80 (m)', fontsize=12)
    ax2.set_ylabel('R95 (m)', fontsize=12)
    ax2.set_title('Constraint Satisfaction Map', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inverse_design_pareto.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Inverse design Pareto plot saved.")


def plot_ejecta_field(P80: float, R95: float,
                       scenario_id: int = 1, output_dir: str = 'outputs/viz'):
    """2D visualization of an ejecta debris field."""
    fig, ax = plt.subplots(figsize=(8, 8))
    theta = np.linspace(0, 2 * np.pi, 360)

    # R95 containment zone
    ax.fill(R95 * np.cos(theta), R95 * np.sin(theta),
             alpha=0.15, color='#ff6b6b', label=f'R95 = {R95:.1f} m')
    ax.plot(R95 * np.cos(theta), R95 * np.sin(theta),
             color='#ff6b6b', lw=2)

    # Simulate debris scatter
    n_debris = 500
    r = np.random.rayleigh(R95 / 3, n_debris)
    t = np.random.uniform(0, 2 * np.pi, n_debris)
    sizes = np.random.exponential(P80, n_debris)
    inside = r < R95

    ax.scatter(r[inside]  * np.cos(t[inside]),
                r[inside]  * np.sin(t[inside]),
                s=np.clip(sizes[inside], 2, 50), alpha=0.6, color='#ffd93d',
                label=f'Fragments (P80={P80:.1f} m)')
    ax.scatter(r[~inside] * np.cos(t[~inside]),
                r[~inside] * np.sin(t[~inside]),
                s=3, alpha=0.3, color='gray')

    ax.scatter(0, 0, s=400, color='white', marker='*', zorder=10,
                label='Impact point')
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(f'Ejecta Field — Scenario {scenario_id}\n'
                  f'P80={P80:.1f} m | R95={R95:.1f} m', fontsize=13)
    ax.legend(); ax.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'ejecta_scenario_{scenario_id}.png'),
                 dpi=150)
    plt.close()

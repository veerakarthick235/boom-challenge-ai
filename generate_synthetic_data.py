"""
boom_challenge/generate_synthetic_data.py
===========================================
Generates realistic synthetic train/test data for the Boom Challenge.
Uses normalized π-group features to produce physically realistic P80 / R95.

Target ranges:
  P80: ~50–200 m  (competition window 96–101 covered)
  R95: ~80–300 m  (competition threshold 175 is meaningful)

Run: python generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
import os

SEED = 42
os.makedirs('data', exist_ok=True)


def simulate_impact(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Raw impact parameters ──────────────────────────────────
    v_i   = rng.uniform(500,   25000, n)           # m/s
    m_i   = rng.uniform(1e6,   1e12,  n)           # kg
    rho_i = rng.uniform(1.5,   8.0,   n)           # g/cm³
    rho_t = rng.uniform(1.5,   3.5,   n)           # g/cm³
    Y     = 10 ** rng.uniform(5, 10,  n)           # Pa  (log-uniform)
    g     = rng.uniform(0.05,  20.0,  n)           # m/s²
    d_i   = rng.uniform(10,    8000,  n)            # m
    theta = rng.uniform(10,    80,    n)            # degrees

    # ── Physics quantities ─────────────────────────────────────
    KE  = 0.5 * m_i * v_i**2                       # Joules  ~10^13–10^24
    rho_t_si = rho_t * 1000                         # kg/m³
    rho_i_si = rho_i * 1000
    sin_t = np.sin(np.radians(theta))

    # Housen-Holsapple  π groups (dimensionless → bounded)
    pi_v  = v_i / (np.sqrt(Y / (rho_t_si + 1e-6)) + 1e-6)   # ~0.003 – 30
    pi_2  = (rho_t_si * g * d_i) / (Y + 1e-6)               # ~1e-7  – 10
    mu_c  = (KE / (Y * d_i**3 + 1e-6))**(1/3)               # ~0.01  – 1000

    # ── Normalized dimensionless features ─────────────────────
    # log-scale each then normalize to [0,1] using known physics bounds
    #   pi_v  ∈ [0.003, 30]  → log10 ∈ [-2.5, 1.5]  range=4
    #   pi_2  ∈ [1e-7,  10]  → log10 ∈ [-7,   1]    range=8
    #   mu_c  ∈ [0.01, 1000] → log10 ∈ [-2,   3]    range=5
    #   g     ∈ [0.05, 20]   → log10 ∈ [-1.3, 1.3]  range=2.6
    #   Y     ∈ [1e5,  1e10] → log10 ∈ [5,   10]    range=5

    npi_v  = (np.log10(np.clip(pi_v,  1e-4, 50))   + 2.5) / 4.0  # [0,1]
    npi_2  = (np.log10(np.clip(pi_2,  1e-8, 20))   + 7.0) / 8.0
    nmu_c  = (np.log10(np.clip(mu_c,  1e-3, 5e3))  + 2.0) / 5.0
    ng     = (np.log10(g + 0.01) + 1.3) / 2.6
    nY     = (np.log10(Y)        - 5.0) / 5.0
    nrho   = (rho_i / (rho_t + 1e-9) - 0.4) / 5.6   # density ratio /normalised

    # ── P80  (fragment size, meters) ──────────────────────────
    # Physics: decreases with pi_v (more shock → finer fragments)
    #          increases with Y    (stronger → coarser)
    #          weakly affected by obliquity
    log10_P80 = (
        2.00           # intercept: 10^2 = 100 m baseline
        - 0.50 * npi_v  # shock → finer
        + 0.30 * nY     # strength → coarser
        - 0.20 * nmu_c  # more cratering efficiency → finer
        + 0.08 * nrho   # density contrast
        + 0.05 * (1 - sin_t)
        + rng.normal(0, 0.04, n)
    )
    P80 = np.clip(10 ** log10_P80, 30, 450)

    # ── R95 (ejecta radius, meters) ────────────────────────────
    # Physics: increases with pi_v (more energy → farther scatter)
    #          decreases with g    (gravity pulls debris back)
    #          slightly increases with obliquity
    log10_R95 = (
        2.20           # intercept: 10^2.2 ≈ 158 m baseline
        + 0.45 * npi_v  # faster/harder → wider
        - 0.30 * ng     # gravity → tighter
        + 0.10 * nmu_c  # cratering size
        + 0.08 * sin_t
        - 0.12 * nY     # stronger target → less scatter
        + rng.normal(0, 0.035, n)
    )
    R95 = np.clip(10 ** log10_R95, 30, 600)

    return pd.DataFrame({
        'v_i': v_i, 'm_i': m_i, 'rho_i': rho_i, 'rho_t': rho_t,
        'Y': Y, 'g': g, 'd_i': d_i, 'theta': theta,
        'P80': P80, 'R95': R95,
    })


# ── Generate ───────────────────────────────────────────────────
print("Generating training data  (5000 samples)...")
train_df = simulate_impact(5000, seed=42)

print("Generating test data (1000 samples)...")
test_df  = simulate_impact(1000, seed=99).drop(columns=['P80','R95'])

train_df.to_csv('data/train.csv', index=False)
test_df.to_csv ('data/test.csv',  index=False)

print(f"\ntrain.csv → {train_df.shape}  |  test.csv → {test_df.shape}")
print("\nTarget summary:")
print(train_df[['P80','R95']].describe().round(2))
print(f"\n  P80 samples in [96,101]: {((train_df.P80>=96)&(train_df.P80<=101)).sum()}")
print(f"  R95 samples ≤ 175:       {(train_df.R95<=175).sum()}")
print(f"\nData saved to data/ ✅")

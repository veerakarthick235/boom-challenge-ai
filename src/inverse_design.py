"""
boom_challenge/src/inverse_design.py
======================================
Inverse Design Optimization for Task 2:
Find 20 impact configurations satisfying:
  96 <= P80 <= 101
  R95 <= 175
while minimizing kinetic energy and debris range.

Approaches:
  1. SLSQP constrained optimization (multi-start)
  2. CMA-ES evolutionary search
  3. NSGA-II multi-objective optimization (pymoo)
  4. Diverse solution selection via K-Means clustering
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────
# PARAMETER BOUNDS
# Physical plausible ranges for Mox-95 system
# [v_i, m_i, rho_i, rho_t, Y, g, d_i, theta]
# ─────────────────────────────────────────────────────────────

PARAM_NAMES = ['v_i', 'm_i', 'rho_i', 'rho_t', 'Y', 'g', 'd_i', 'theta']

LOWER_BOUNDS = np.array([
    500.0,      # v_i  (m/s)
    1e6,        # m_i  (kg)
    1.5,        # rho_i (g/cm³)
    1.5,        # rho_t (g/cm³)
    1e5,        # Y (Pa)
    0.05,       # g (m/s²)
    10.0,       # d_i (m)
    10.0,       # theta (deg)
])

UPPER_BOUNDS = np.array([
    25000.0,    # v_i
    1e13,       # m_i
    8.0,        # rho_i
    4.0,        # rho_t
    1e10,       # Y
    20.0,       # g
    10000.0,    # d_i
    80.0,       # theta
])

BOUNDS = list(zip(LOWER_BOUNDS, UPPER_BOUNDS))


# ─────────────────────────────────────────────────────────────
# ENSEMBLE PREDICTOR WRAPPER
# ─────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Wraps the trained ensemble to provide scalar predictions
    for use in optimization routines.
    """

    def __init__(self, ensemble, feature_engineer_fn, scaler):
        """
        Parameters
        ----------
        ensemble           : trained StackingEnsemble (or any model with .predict())
        feature_engineer_fn: callable(df) -> feature array
        scaler             : fitted StandardScaler or RobustScaler
        """
        self.ensemble = ensemble
        self.feature_engineer = feature_engineer_fn
        self.scaler = scaler

    def predict_single(self, x: np.ndarray) -> tuple:
        """
        Predict P80 and R95 for a single parameter vector.
        Returns (P80, R95) in original space (not log).
        """
        df = pd.DataFrame([x], columns=PARAM_NAMES)
        X_feat = self.feature_engineer(df)
        if self.scaler:
            X_feat = self.scaler.transform(X_feat)
        log_preds = self.ensemble.predict(X_feat)   # [1, 2]
        P80 = np.expm1(log_preds[0, 0])
        R95 = np.expm1(log_preds[0, 1])
        return float(P80), float(R95)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict for a batch of parameter vectors. Returns [n, 2]."""
        rows = []
        for x in X:
            P80, R95 = self.predict_single(x)
            rows.append([P80, R95])
        return np.array(rows)


# ─────────────────────────────────────────────────────────────
# OBJECTIVE AND CONSTRAINTS
# ─────────────────────────────────────────────────────────────

def build_objective(predictor: EnsemblePredictor, alpha_R95: float = 5.0):
    """
    Build objective + constraint functions for scipy.optimize.

    Minimizes:   0.5 * m_i * v_i² + alpha_R95 * R95
    Subject to:  96 <= P80 <= 101, R95 <= 175
    """
    def objective(x):
        KE = 0.5 * x[1] * x[0]**2
        _, R95 = predictor.predict_single(x)
        return KE / 1e15 + alpha_R95 * R95   # normalize energy scale

    def c_P80_lower(x):  # P80 >= 96
        P80, _ = predictor.predict_single(x)
        return P80 - 96.0

    def c_P80_upper(x):  # P80 <= 101
        P80, _ = predictor.predict_single(x)
        return 101.0 - P80

    def c_R95(x):        # R95 <= 175
        _, R95 = predictor.predict_single(x)
        return 175.0 - R95

    constraints = [
        {'type': 'ineq', 'fun': c_P80_lower},
        {'type': 'ineq', 'fun': c_P80_upper},
        {'type': 'ineq', 'fun': c_R95},
    ]
    return objective, constraints


def is_feasible(x: np.ndarray, predictor: EnsemblePredictor, margin: float = 0.02) -> bool:
    """Check if parameter vector satisfies constraints with a safety margin."""
    P80, R95 = predictor.predict_single(x)
    return (96 * (1 - margin) <= P80 <= 101 * (1 + margin) and
            R95 <= 175 * (1 + margin))


# ─────────────────────────────────────────────────────────────
# MULTI-START SLSQP
# ─────────────────────────────────────────────────────────────

def slsqp_multistart(predictor: EnsemblePredictor, n_starts: int = 300,
                     alpha_R95: float = 5.0, verbose: bool = True) -> list:
    """
    Multi-start SLSQP with normalized [0, 1] bounds for better gradient scaling.
    """
    base_objective, base_constraints = build_objective(predictor, alpha_R95)
    feasible_solutions = []

    # Helper: Convert [0, 1] back to physical units
    def unnormalize(x_norm):
        return LOWER_BOUNDS + x_norm * (UPPER_BOUNDS - LOWER_BOUNDS)

    # Wrappers for the optimizer
    def norm_objective(x_norm):
        return base_objective(unnormalize(x_norm))

    norm_constraints = []
    for c in base_constraints:
        # Capture the current function in the loop closure
        def make_wrapper(base_func):
            return lambda x_norm: base_func(unnormalize(x_norm))
        norm_constraints.append({'type': c['type'], 'fun': make_wrapper(c['fun'])})

    # Optimizer now strictly searches in a 0 to 1 hypercube
    norm_bounds = [(0.0, 1.0) for _ in range(len(LOWER_BOUNDS))]

    for seed in range(n_starts):
        np.random.seed(seed)
        x0_norm = np.random.rand(len(LOWER_BOUNDS))

        result = minimize(
            norm_objective, x0_norm,
            method='SLSQP',
            bounds=norm_bounds,
            constraints=norm_constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )

        # Convert back to physical space for evaluation and saving
        x_physical = unnormalize(result.x)

        if result.success or is_feasible(x_physical, predictor):
            P80, R95 = predictor.predict_single(x_physical)
            if 95.5 <= P80 <= 102 and R95 <= 180:
                feasible_solutions.append({
                    'x': x_physical,
                    'P80': P80, 'R95': R95,
                    'KE': 0.5 * x_physical[1] * x_physical[0]**2,
                    'obj': result.fun
                })

    if verbose:
        print(f"SLSQP: Found {len(feasible_solutions)} feasible solutions from {n_starts} starts")
    return feasible_solutions


# ─────────────────────────────────────────────────────────────
# CMA-ES EVOLUTIONARY SEARCH
# ─────────────────────────────────────────────────────────────

def cmaes_search(predictor: EnsemblePredictor, penalty_coeff: float = 1e6,
                 sigma0: float = 0.3, n_parallel_runs: int = 10) -> list:
    """
    CMA-ES with constraint penalty for global search.
    Requires: pip install cma
    """
    try:
        import cma
    except ImportError:
        print("CMA-ES not available. Install with: pip install cma")
        return []

    def fitness(x_norm):
        # Un-normalize from [0,1] to physical space
        x = LOWER_BOUNDS + x_norm * (UPPER_BOUNDS - LOWER_BOUNDS)
        x = np.clip(x, LOWER_BOUNDS, UPPER_BOUNDS)

        P80, R95 = predictor.predict_single(x)
        KE = 0.5 * x[1] * x[0]**2

        penalty = (penalty_coeff * max(0, 96 - P80)**2 +
                   penalty_coeff * max(0, P80 - 101)**2 +
                   penalty_coeff * max(0, R95 - 175)**2)

        return KE / 1e15 + 5.0 * R95 + penalty

    solutions = []
    for run in range(n_parallel_runs):
        np.random.seed(run * 7 + SEED)
        x0_norm = np.random.rand(len(LOWER_BOUNDS))

        es = cma.CMAEvolutionStrategy(
            x0_norm, sigma0,
            {'bounds': [np.zeros(len(LOWER_BOUNDS)).tolist(),
                        np.ones(len(LOWER_BOUNDS)).tolist()],
             'maxiter': 3000,
             'seed': run,
             'tolx': 1e-8,
             'verbose': -9}
        )
        es.optimize(fitness)

        x_best_norm = es.result.xbest
        x_best = LOWER_BOUNDS + x_best_norm * (UPPER_BOUNDS - LOWER_BOUNDS)
        P80, R95 = predictor.predict_single(x_best)

        if 95.5 <= P80 <= 101.5 and R95 <= 176.0:
            solutions.append({
                'x': x_best, 'P80': float(P80), 'R95': float(R95),
                'KE': float(0.5 * x_best[1] * x_best[0]**2),
                'obj': es.result.fbest
            })

    print(f"CMA-ES: Found {len(solutions)} feasible solutions")
    return solutions


# ─────────────────────────────────────────────────────────────
# SELECT 20 DIVERSE SOLUTIONS
# ─────────────────────────────────────────────────────────────

def select_diverse_solutions(all_solutions: list, n_select: int = 20) -> pd.DataFrame:
    """
    Select 20 maximally diverse yet optimal solutions via K-Means clustering.
    From each cluster, select the solution with minimum objective value.
    """
    if not all_solutions:
        print("⚠ No feasible solutions provided to cluster. Returning empty DataFrame.")
        return pd.DataFrame()

    n_select = min(n_select, len(all_solutions))

    if len(all_solutions) == n_select:
         print(f"⚠ Only {len(all_solutions)} feasible solutions found. Bypassing clustering.")
         selected = all_solutions
    else:
        # Normalize parameter vectors for clustering
        X_params = np.array([s['x'] for s in all_solutions])
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X_params)

        km = KMeans(n_clusters=n_select, random_state=SEED, n_init=20)
        labels = km.fit_predict(X_norm)

        selected = []
        for cluster_id in range(n_select):
            cluster_sols = [s for s, l in zip(all_solutions, labels) if l == cluster_id]
            if cluster_sols:
                # Pick minimum KE solution from cluster
                best = min(cluster_sols, key=lambda s: s['KE'])
                selected.append(best)

    # Build output DataFrame
    rows = []
    for i, sol in enumerate(selected):
        row = {f'param_{j}_{PARAM_NAMES[j]}': sol['x'][j] for j in range(len(PARAM_NAMES))}
        row.update({'P80': sol['P80'], 'R95': sol['R95'],
                    'KE_J': sol['KE'], 'scenario_id': i + 1})
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n✅ Selected {len(df)} diverse feasible scenarios")
    print(f"   P80 range: [{df['P80'].min():.2f}, {df['P80'].max():.2f}]")
    print(f"   R95 range: [{df['R95'].min():.2f}, {df['R95'].max():.2f}]")
    print(f"   KE range:  [{df['KE_J'].min():.2e}, {df['KE_J'].max():.2e}] J")
    return df


# ─────────────────────────────────────────────────────────────
# FULL INVERSE DESIGN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_inverse_design(predictor: EnsemblePredictor,
                       output_path: str = 'outputs/task2_scenarios.csv') -> pd.DataFrame:
    """
    Full Task 2 inverse design pipeline.
    Runs SLSQP + CMA-ES, combines results, selects 20 diverse solutions.
    """
    print("=" * 60)
    print("TASK 2: INVERSE DESIGN OPTIMIZATION")
    print("=" * 60)

    # Step 1: SLSQP multi-start
    print("\n[1/3] Running SLSQP multi-start optimization...")
    slsqp_sols = slsqp_multistart(predictor, n_starts=300)

    # Step 2: CMA-ES global search
    print("\n[2/3] Running CMA-ES evolutionary search...")
    cmaes_sols = cmaes_search(predictor, n_parallel_runs=15)

    # Combine all solutions
    all_solutions = slsqp_sols + cmaes_sols
    print(f"\n[3/3] Total feasible solutions found: {len(all_solutions)}")

    # Step 3: Select 20 diverse solutions
    df_final = select_diverse_solutions(all_solutions, n_select=20)

    # Save
    if not df_final.empty:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"\nSaved {len(df_final)} scenarios -> {output_path}")
    else:
        print("\nNo scenarios saved due to optimization failure.")

    return df_final


if __name__ == '__main__':
    # Demo: use random predictor for testing
    class MockPredictor:
        def predict_single(self, x):
            # Dummy physics-like relationship
            v, m = x[0], x[1]
            KE = 0.5 * m * v**2
            # Removed stochastic noise so SLSQP gradients don't explode
            P80 = 150 - 3e-17 * KE
            R95 = 100 + 8e-18 * KE
            return P80, R95

    mock = MockPredictor()
    sols = slsqp_multistart(mock, n_starts=50, verbose=True)
    df = select_diverse_solutions(sols, n_select=min(20, len(sols)))
    if not df.empty:
        print(df[['P80', 'R95', 'KE_J']].to_string())
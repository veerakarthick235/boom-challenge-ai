# 💥 Boom: Trajectory Unknown Challenge
### Physics-Informed Machine Learning for Asteroid Impact Ejecta Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.ai)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-green.svg)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌌 Problem Overview

An asteroid strikes the surface of a planet in the **Mox-95 stellar system**. Our goal is to predict:
- **P80** — 80th percentile fragment size from the impact debris field
- **R95** — radius containing 95% of all ejecta

This is both a **forward prediction problem** (impact params → debris outcomes) and an **inverse design problem** (find configurations that satisfy debris constraints).

---

## 🏗 Repository Structure

```
boom_challenge/
├── data/
│   ├── train.csv                  # Training dataset
│   └── test.csv                   # Test dataset
├── src/
│   ├── feature_engineering.py     # Physics-based feature generation
│   ├── ensemble.py                # Stacking ensemble + adversarial validation
│   ├── inverse_design.py          # Task 2: constraint optimization
│   ├── visualization.py           # Plotting utilities
│   └── models/
│       ├── pinn_model.py          # PyTorch Physics-Informed Neural Network
│       └── gbm_models.py          # XGBoost + LightGBM with Optuna HPO
├── outputs/
│   ├── task1_submission.csv       # Forward prediction results
│   ├── task2_scenarios.csv        # 20 inverse design scenarios
│   ├── models/                    # Saved model artifacts
│   └── eda/                       # Visualization outputs
├── notebooks/
│   └── eda_and_analysis.ipynb    # Exploratory analysis notebook
├── train.py                       # Main training pipeline
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🧪 Key Technical Approach

### Physics-Informed Feature Engineering
We derive dimensionless **Buckingham π groups** from the Housen-Holsapple impact scaling laws:

| Feature | Formula | Physical Meaning |
|---|---|---|
| `pi_v` | v / √(Y/ρ_t) | Shock velocity ratio |
| `pi_2` | ρ_t × g × d / Y | Gravity-scaling param |
| `mu_crater` | (KE/Y)^(1/3) × (ρ_i/ρ_t)^(1/3) | Cratering efficiency |
| `E_spec` | KE / (ρ_t × d³) | Specific impact energy |

### Model Stack
```
┌─ XGBoost  ─┐ ┌─ LightGBM ─┐ ┌─ PyTorch PINN ─┐
│ Optuna HPO │ │ Optuna HPO │ │ Physics Loss   │
└────────────┘ └────────────┘ └────────────────┘
         \              |              /
          ┌─────────────────────────┐
          │  Ridge Meta-Learner     │  ← trained on OOF
          │  (Stacking Ensemble)    │
          └─────────────────────────┘
```

### Physics-Informed Neural Network (PINN)
The PINN enforces **monotonicity constraints** via automatic differentiation:
- dP80/d(log_KE) ≤ 0 — higher energy → finer fragments
- dR95/d(log_KE) ≥ 0 — higher energy → wider scatter

### Inverse Design (Task 2)
Multi-strategy optimization to find 20 configurations with `96 ≤ P80 ≤ 101` and `R95 ≤ 175`:
1. **SLSQP** — gradient-based constrained optimization (500 multi-starts)
2. **CMA-ES** — evolutionary global search
3. **K-Means** — diverse solution selection from all feasible candidates

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourname/boom-challenge
cd boom-challenge
pip install -r requirements.txt
```

### Training (Task 1)
```bash
# Basic training
python train.py --data_dir data/ --output_dir outputs/

# With Optuna hyperparameter optimization
python train.py --data_dir data/ --output_dir outputs/ --run_hpo

# Full pipeline including Task 2
python train.py --data_dir data/ --output_dir outputs/ --run_hpo --run_task2
```

### View experiment tracking
```bash
mlflow ui
# Navigate to http://localhost:5000
```

---

## 📊 Expected Performance

| Metric | P80 | R95 |
|---|---|---|
| Target RMSE | < 5.0 | < 10.0 |
| Target R² | > 0.92 | > 0.90 |
| Physics constraints satisfied | 100% | 100% |

---

## 🔬 Physics Background

The solution is grounded in **Grady-Kipp fragmentation theory** and **Housen-Holsapple π-scaling**:

- Fragment size distributions follow power laws governed by strain-rate and material strength
- Ejecta radii scale with impact energy via: `R95 ∝ KE^γ × g^δ × Y^ε`
- Training in log-space naturally captures these power-law relationships

---

## 📁 Output Files

| File | Description |
|---|---|
| `outputs/task1_submission.csv` | Task 1: P80 and R95 for all test samples |
| `outputs/task2_scenarios.csv` | Task 2: 20 feasible impact configurations |
| `outputs/models/` | Saved XGBoost, LightGBM, PINN checkpoint |
| `outputs/eda/` | EDA plots, SHAP summaries, residual diagnostics |

---

## 🛡 Reproducibility

All random seeds are fixed via:
```python
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
```

---

## 📚 References

1. Housen, K.R. & Holsapple, K.A. (2011). *Ejecta from impact craters*. Icarus.
2. Grady, D.E. & Kipp, M.E. (1980). *Continuum modelling of explosive fracture*. Int. J. Rock Mech.
3. Raissi, M. et al. (2019). *Physics-informed neural networks*. J. Comput. Physics.

---

*Built for the Boom: Trajectory Unknown Challenge — Mox-95 Impact Physics Track*

# 💥 Boom: Trajectory Unknown Challenge
### Physics-Informed Machine Learning for Asteroid Impact Ejecta Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.ai)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-green.svg)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌌 Problem Overview

An asteroid strikes the surface of a planet in the **Mox-95 stellar system**. Our goal is to predict six critical debris characteristics, primarily focusing on:
- **P80** — 80th percentile fragment size from the impact debris field
- **R95** — radius containing 95% of all ejecta
- *Additional Targets:* `fines_frac`, `oversize_frac`, `R50_fines`, `R50_oversize`

This is a dual-objective challenge: a **forward prediction problem** (impact params → debris outcomes) and an **inverse design problem** (finding optimal configurations that strictly satisfy P80 and R95 bounds).

---

## 🏗 Repository Structure

```text
boom_challenge/
├── data/
│   ├── train.csv                  # Official training features
│   ├── train_labels.csv           # Official training targets
│   ├── test.csv                   # Official test features
│   └── prediction_submission_template.csv
├── src/
│   ├── feature_engineering.py     # Dimensionless π-group scaling logic
│   ├── ensemble.py                # Stacking Regressor & OOF logic
│   ├── inverse_design.py          # Smart Monte Carlo Optimizer (Task 2)
│   ├── visualization.py           # Result diagnostics
│   └── models/
│       ├── pinn_model.py          # Physics-Informed Neural Network
│       └── gbm_models.py          # Optuna-tuned XGBoost & LightGBM
├── outputs/
│   ├── task1_submission.csv       # Final Forward Predictions
│   ├── task2_scenarios.csv        # 20 Optimized Inverse Scenarios
│   ├── models/                    # Serialized model weights
│   └── eda/                       # Data analysis plots
├── train.py                       # Main pipeline execution script
├── requirements.txt               # Dependency list
├── mlflow.db                      # Local experiment tracking database
└── README.md
---

## 🧪 Key Technical Approach

### Physics-Informed Feature Engineering

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
          │  Ridge Meta-Learner     │
          │  (Stacking Ensemble)    │
          └─────────────────────────┘
```

### Inverse Design (Task 2)

Smart Monte Carlo Simulator:
1. Generates 500,000 random impact scenarios
2. Evaluates using tuned XGBoost ensemble
3. Selects optimal configurations via nearest-neighbor filtering

---

## 🚀 Quick Start

```bash
git clone https://github.com/veerakarthick235/boom-challenge-ai.git
cd boom-challenge-ai
pip install -r requirements.txt
```

```bash
python train.py --data_dir data/ --output_dir outputs/ --run_hpo --run_task2
```

---

## 📊 Performance Metrics

| Target | R² Score | MAE | MAPE |
|---|---|---|---|
| **P80** | **0.975** | 7.93 | 4.60% |
| **R95** | **0.920** | 39.60 | 17.15% |
| **Oversize Frac** | **0.990** | 0.026 | 21.36% |

---

## ⭐ Key Highlights

- Physics-informed ML pipeline
- Stacking ensemble with PINN + GBMs
- MLflow experiment tracking
- Monte Carlo inverse design engine

---

## 🛡 Reproducibility

```python
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
```

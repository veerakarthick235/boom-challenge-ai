# 💥 Boom: Trajectory Unknown Challenge

## Physics-Informed Machine Learning for Asteroid Impact Ejecta Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.ai)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-green.svg)](https://mlflow.org)

---

## 🌌 Problem Overview

An asteroid impacts the surface of a planet in the Mox-95 stellar system.

The objective is to predict six critical ejecta characteristics:

- **P80** — 80th percentile fragment size
- **R95** — Radius containing 95% of ejecta
- **fines_frac** — Fraction of fine particles
- **oversize_frac** — Fraction of large fragments
- **R50_fines** — Median radius of fines
- **R50_oversize** — Median radius of oversize fragments

The challenge consists of:

### Task 1 — Forward Prediction
Predict ejecta characteristics from asteroid impact parameters.

### Task 2 — Inverse Design
Generate impact scenarios that satisfy specified ejecta constraints.

---

## 🏗 Repository Structure

```text
boom_challenge/
├── data/
├── src/
│   ├── feature_engineering.py
│   ├── ensemble.py
│   ├── inverse_design.py
│   ├── visualization.py
│   └── models/
│       ├── pinn_model.py
│       └── gbm_models.py
├── outputs/
├── train.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Physics-Informed Feature Engineering

The solution incorporates physically meaningful features derived from impact mechanics and crater-scaling relationships.

### Example Derived Features

| Feature | Description |
|----------|-------------|
| pi_v | Shock velocity ratio |
| pi_2 | Gravity scaling parameter |
| mu_crater | Crater efficiency estimate |
| E_spec | Specific impact energy |

These engineered variables help encode domain knowledge directly into the learning process.

---

## 🤖 Model Architecture

The solution uses a three-model ensemble architecture.

### Base Models

#### XGBoost
Primary gradient boosting learner optimized using Optuna.

#### LightGBM
Complementary boosting model for improved generalization.

#### Physics-Informed Neural Network (PINN)
Neural network trained with additional physics-based constraints to encourage physically consistent predictions.

---

### Hyperparameter Optimization

Optuna is used for automated hyperparameter search and model tuning.

---

### Ensemble Strategy

Predictions from individual models are combined using a validation-driven weighted ensemble.

```text
Final Prediction
= (0.60 × XGBoost)
+ (0.30 × LightGBM)
+ (0.10 × PINN)
```

The ensemble weights were selected based on validation performance and leaderboard-oriented optimization.

> Note:
> This implementation uses weighted blending rather than a traditional stacked meta-learner.

---

## 🧠 Inverse Design Engine

The inverse-design pipeline searches for asteroid impact configurations satisfying target constraints.

Key components:

- Large-scale Monte Carlo candidate generation
- Constraint-based filtering
- Diversity-aware candidate selection
- Energy minimization strategy
- Scenario ranking

---

## 📊 Performance Metrics

| Target | R² Score | MAE |
|----------|----------|----------|
| P80 | 0.975 | 7.93 |
| R95 | 0.920 | 39.60 |
| fines_frac | 0.946 | 0.007 |
| oversize_frac | 0.990 | 0.026 |
| R50_fines | 0.898 | 50.03 |
| R50_oversize | 0.857 | 23.18 |

---

## 🏆 Competition Highlights

- Physics-informed ML workflow
- Ensemble learning architecture
- Automated hyperparameter optimization
- MLflow experiment tracking
- Forward prediction pipeline
- Inverse-design optimization engine
- Reproducible training setup

---

## 🚀 Installation

```bash
git clone https://github.com/veerakarthick235/boom-challenge-ai.git
cd boom-challenge-ai
pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python train.py --data_dir data/ --output_dir outputs/
```

Hyperparameter optimization:

```bash
python train.py --run_hpo
```

Full pipeline:

```bash
python train.py --run_hpo --run_task2
```

---

## 📈 MLflow Tracking

```bash
python -m mlflow ui
```

Launch dashboard:

```text
http://localhost:5000
```

---

## 📂 Outputs

### Task 1

`task1_submission.csv`

Contains predictions for:

- P80
- R95
- fines_frac
- oversize_frac
- R50_fines
- R50_oversize

### Task 2

`task2_scenarios.csv`

Contains optimized inverse-design impact scenarios.

---

## 🔁 Reproducibility

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

---

## 📚 References

- Housen & Holsapple (2011) – Impact ejecta scaling
- Grady & Kipp (1980) – Fragmentation theory
- Raissi et al. (2019) – Physics-Informed Neural Networks

---

## 🚀 Final Note

Built for the Boom: Trajectory Unknown Challenge.

This project combines:
- Physics-informed learning
- Ensemble modeling
- Optimization-driven inverse design

to create a robust asteroid impact prediction framework.

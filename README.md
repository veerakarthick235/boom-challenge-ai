# 💥 Boom: Trajectory Unknown Challenge

### Physics-Informed Machine Learning for Asteroid Impact Ejecta Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.ai)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-green.svg)](https://mlflow.org)

---

## 🌌 Problem Overview

An asteroid strikes the surface of a planet in the **Mox-95 stellar system**.

We predict **six critical ejecta characteristics**:

* **P80** — 80th percentile fragment size
* **R95** — radius containing 95% of ejecta
* **fines_frac** — fraction of fine particles
* **oversize_frac** — fraction of large fragments
* **R50_fines** — median radius of fines
* **R50_oversize** — median radius of oversize fragments

This is a dual challenge:

* **Forward Prediction** → impact parameters → debris outcomes
* **Inverse Design** → find optimal impact conditions satisfying constraints

---

## 🏗 Repository Structure

```
boom_challenge/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── feature_engineering.py
│   ├── ensemble.py
│   ├── inverse_design.py
│   ├── visualization.py
│   └── models/
│       ├── pinn_model.py
│       └── gbm_models.py
├── outputs/
│   ├── task1_submission.csv
│   ├── task2_scenarios.csv
│   ├── models/
│   └── eda/
├── train.py
├── requirements.txt
└── README.md
```

---

## 🧪 Key Technical Approach

### ⚙️ Physics-Informed Feature Engineering

We derive dimensionless **Buckingham π groups**:

| Feature     | Formula                        | Meaning              |
| ----------- | ------------------------------ | -------------------- |
| `pi_v`      | v / √(Y/ρ_t)                   | Shock velocity ratio |
| `pi_2`      | ρ_t × g × d / Y                | Gravity scaling      |
| `mu_crater` | (KE/Y)^(1/3) × (ρ_i/ρ_t)^(1/3) | Crater efficiency    |
| `E_spec`    | KE / (ρ_t × d³)                | Specific energy      |

---

### 🤖 Model Architecture

```
Our pipeline employs a **Three-Tier Elite Stacking Ensemble**:

1.  **Base Layer (Diverse Learners):**
    * **XGBoost:** Gradient boosting optimized via Optuna (Primary Learner).
    * **LightGBM:** Fast gradient boosting optimized for broad target coverage.
    * **PyTorch PINN:** A Physics-Informed Neural Network enforcing energy-consistency laws.
2.  **Optimization Layer:**
    * **Optuna HPO:** Automated Bayesian search for optimal hyperparameters.
3.  **Meta Layer (Hardcoded Elite Blend):**
    * Instead of a variable meta-learner, we utilize a mathematically derived optimal weighted blend:
    * `Final = (0.60 * XGB) + (0.30 * LGBM) + (0.10 * PINN)`
```

* XGBoost → primary learner
* LightGBM → complementary patterns
* PINN → physics constraints
* Ridge → stacking ensemble

---

### 🧠 Inverse Design Engine

High-throughput optimization:

* 🔹 500,000 Monte Carlo samples
* 🔹 Constraint filtering (P80, R95 bounds)
* 🔹 Diversity selection via clustering
* 🔹 Energy minimization for optimal solutions

---

## 🏆 Competition Strategy

* Optimized for official weighted scoring
* Strong focus on:

  * **P80 (30%)**
  * **R95 (20%)**
* Specialized handling for:

  * Fractional outputs
  * Distribution-based targets
* Achieves:

  * **20/20 valid inverse design solutions**
  * **Low-impact energy optimization**
  * **High diversity across scenarios**

---

## 🚀 Quick Start

### 🔧 Installation

```bash
git clone https://github.com/veerakarthick235/boom-challenge-ai.git
cd boom-challenge-ai
pip install -r requirements.txt
```

---

### 🏋️ Training

```bash
# Basic training
python train.py --data_dir data/ --output_dir outputs/

# With hyperparameter tuning
python train.py --data_dir data/ --output_dir outputs/ --run_hpo

# Full pipeline (Task 1 + Task 2)
python train.py --data_dir data/ --output_dir outputs/ --run_hpo --run_task2
```

---

### 📊 MLflow Tracking

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open: http://localhost:5000

---

## 📊 Performance Metrics

| Target        | R² Score  | MAE   | MAPE   |
| ------------- | --------- | ----- | ------ |
| **P80**       | **0.975** | 7.93  | 4.60%  |
| **R95**       | **0.920** | 39.60 | 17.15% |
| fines_frac    | 0.946     | 0.007 | 20.9%  |
| oversize_frac | 0.990     | 0.026 | 21.3%  |
| R50_fines     | 0.898     | 50.03 | 17.0%  |
| R50_oversize  | 0.857     | 23.18 | 18.9%  |

---

## 📁 Outputs

| File                   | Description                   |
| ---------------------- | ----------------------------- |
| `task1_submission.csv` | Forward predictions           |
| `task2_scenarios.csv`  | 20 optimized impact scenarios |
| `models/`              | Saved models                  |
| `eda/`                 | Plots & analysis              |

---

## ⭐ Key Highlights

* Physics-informed ML pipeline
* Advanced stacking ensemble
* Optuna hyperparameter tuning
* MLflow experiment tracking
* High-performance inverse design
* Competition-optimized solution

---

## 🛡 Reproducibility

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

## 📚 References

* Housen & Holsapple (2011) — Impact ejecta scaling
* Grady & Kipp (1980) — Fragmentation theory
* Raissi et al. (2019) — Physics-Informed Neural Networks

---

## 🚀 Final Note

Built for **Boom: Trajectory Unknown Challenge**
Focused on **physics + ML + optimization synergy**

👉 Designed for **top leaderboard performance**

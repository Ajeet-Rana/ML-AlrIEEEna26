# ML Challenge — IEEE SB GEHU
## Fault Detection System | Approach & Methodology

---

## 📌 Problem Statement

Binary classification task to predict whether an embedded device is operating normally or experiencing a fault condition based on 47 numerical sensor features.

| | |
|---|---|
| **Input** | 47 numerical features (F01–F47) |
| **Target** | Class 0 = Normal, Class 1 = Faulty |
| **Train Size** | 43,776 rows |
| **Test Size** | 10,944 rows |
| **Metrics** | Accuracy & F1-Score |

---

## 🔍 Step 1 — Exploratory Data Analysis

- **Class Distribution:** 60.5% Normal vs 39.5% Faulty — moderately imbalanced
- **Missing Values:** None found — clean dataset
- **Duplicate Rows:** 738 exact duplicate rows detected
- **Feature Count:** 47 numerical features, no categorical variables

---

## 🧹 Step 2 — Data Cleaning

| Action | Reason |
|---|---|
| Dropped 738 duplicate rows | Exact copies bias model training |
| Dropped F20 | Near-zero variance (std = 0.000179) — carries no information |
| Dropped F24, F25, F26, F27, F28, F29 | Correlation > 0.95 with F04–F09 — redundant features |

**Final feature count after cleaning: 40 features**

---

## 🏆 Step 3 — Model Selection

Trained 6 models on cleaned data to justify final model choice:

| Model | Accuracy | F1 Score | Reason for Rejection |
|---|---|---|---|
| Logistic Regression | ~86% | ~84% | Assumes linear relationships |
| Decision Tree | ~93% | ~92% | Overfits, no boosting |
| SVM | ~90% | ~89% | Slow, struggles with 40 features |
| Random Forest | ~96% | ~95% | Good but no inter-tree learning |
| **XGBoost** | **98.27%** | **97.84%** | ✅ Selected |
| **LightGBM** | **98.29%** | **97.87%** | ✅ Selected |

**Why XGBoost & LightGBM?**
- Built for tabular/numerical data
- Gradient boosting — each tree learns from previous tree's mistakes
- Built-in regularization prevents overfitting
- Immune to outliers (tree-based splits)
- Proven winners in tabular ML competitions

---

## ⚙️ Step 4 — Baseline Training

Both models trained with default parameters and `scale_pos_weight` to handle class imbalance:

```
scale_pos_weight = 26465 / 17311 ≈ 1.53
```

| Model | Accuracy | F1 Score |
|---|---|---|
| XGBoost (baseline) | 98.27% | 97.84% |
| LightGBM (baseline) | 98.29% | 97.87% |

---

## ⚡ Step 5 — Hyperparameter Tuning with Optuna

Used **Optuna TPE Sampler** (Bayesian Optimization) with 30 trials per model to find optimal hyperparameters.

**Parameters tuned:**
- `n_estimators` — number of trees
- `max_depth` — tree depth
- `learning_rate` — contribution of each tree
- `subsample` — row sampling ratio
- `colsample_bytree` — feature sampling ratio
- `min_child_weight` / `num_leaves` — leaf node constraints
- `reg_alpha`, `reg_lambda` — L1/L2 regularization

| Model | Accuracy | F1 Score | Improvement |
|---|---|---|---|
| XGBoost (tuned) | 98.84% | 98.55% | +0.71% F1 |
| LightGBM (tuned) | 98.70% | 98.37% | +0.50% F1 |

---

## 🤝 Step 6 — Ensemble

Combined both tuned models using **Soft Voting** (averaging predicted probabilities):

```
Final Probability = (XGBoost_prob + LightGBM_prob) / 2
Final Class = 1 if probability >= 0.5 else 0
```

| Approach | Accuracy | F1 Score |
|---|---|---|
| XGBoost alone | 98.84% | 98.55% |
| Equal Ensemble (50/50) | 98.84% | 98.55% |
| Weighted Ensemble (60/40) | 98.82% | 98.52% |

---

## 📊 Final Results

| Metric | Score |
|---|---|
| **Validation Accuracy** | **98.84%** |
| **Validation F1 Score** | **98.55%** |
| True Normal (correct) | 5,087 |
| True Faulty (correct) | 3,372 |
| False Alarms | 59 |
| Missed Faults | 90 |

---

## 🔑 Top Features (by XGBoost Importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | F01 | 0.0888 |
| 2 | F19 | 0.0607 |
| 3 | F09 | 0.0548 |
| 4 | F36 | 0.0431 |
| 5 | F22 | 0.0408 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | Preprocessing & evaluation |
| XGBoost | Primary model |
| LightGBM | Secondary model |
| Optuna | Hyperparameter tuning |
| Matplotlib & Seaborn | Visualization |
| Google Colab | Development environment |

---

*ML Challenge — IEEE SB GEHU | Binary Fault Detection*

# ToolLife ML – Intelligent Machining Tool Life Prediction System

A production-grade industrial analytics and predictive modeling application for machining operations. Built with **Streamlit**, **Scikit-Learn**, and **Statsmodels**, this platform forecasts lathe cutting tool lifespan ($Y$, in hours) based on cutting velocities, tool metallurgy, feed rate, depth of cut, coolant application, and workpiece hardness.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Dataset & Variables](#dataset--variables)
- [Machine Learning & Statistical Methodology](#machine-learning--statistical-methodology)
- [The Five Assignment Solutions](#the-five-assignment-solutions)

---

## 🔬 Project Overview

Tool life prediction is critical in computer numerical control (CNC) machining and manufacturing metrology. Cutting tools degrade under severe mechanical friction and thermal softening (following Taylor's classical tool life relationship $V \cdot T^n = C$). 

This platform leverages empirical data from 100 industrial turning cycles to provide:
1. **Interactive Real-Time Tool Life Inference** with sensitivity sweeps and domain boundary guardrails.
2. **Exploratory Data Metrology & Diagnostics** (distributions, missing values, correlation matrices).
3. **Parametric Ordinary Least Squares (OLS) Regression Analysis** comparing the full 6-variable model against the reduced speed + tool grade model.
4. **Machine Learning Model Benchmark** evaluating Ridge Regularization against Extra Trees, Gradient Boosting, and Random Forest.
5. **Academic & Examination Solutions** for all five assignment questions directly verified against the original workbook.

---

## 🏛️ System Architecture

The application is structured into clean, decoupled, modular components:

```
TOOL LIFE/
│
├── app.py                         # Main Streamlit application entrypoint & routing
├── requirements.txt               # Complete Python dependencies
├── README.md                      # Documentation & operational guide
├── TOOL_LIFE_DATA_100.xlsx        # Original 100-run experimental workbook
├── TOOL_LIFE_DATA_100_ANSWERED.xlsx # Workbook containing answers
│
├── modules/                       # Modular business logic & views
│   ├── __init__.py
│   ├── data_loader.py             # Resilient data, metadata, and model loading
│   ├── ui_theme.py                # Industrial SCADA/metrology design system & Plotly layouts
│   ├── prediction_view.py         # Real-time tool life inference & sensitivity curves
│   ├── analytics_view.py          # Descriptive statistics, missingness & correlations
│   ├── regression_view.py         # Full & reduced OLS regression equations & stats
│   ├── model_eval_view.py         # 4-model cross-validation benchmarking & pipeline flow
│   └── assignment_qa_view.py      # The five assignment questions and calculated answers
│
├── outputs/                       # Verified precomputed artifacts & models
│   ├── tool_life_model.joblib     # Production trained Ridge regression pipeline
│   ├── summary.json               # Precomputed summary metrics
│   ├── model_comparison.csv       # 5-fold CV and test set evaluation results
│   ├── full_ols_summary.txt       # Statsmodels output for full OLS model
│   ├── reduced_ols_summary.txt    # Statsmodels output for reduced OLS model
│   ├── full_ols_coefficients.csv  # Full model parameter estimates
│   ├── reduced_ols_coefficients.csv # Reduced model parameter estimates
│   ├── missing_values.csv         # Column-by-column missing value counts
│   ├── correlations.csv           # Pearson correlation with target Y
│   ├── tool_life_by_tool_type.csv # Tool A vs. Tool B moments
│   ├── tool_life_by_coolant.csv   # Coolant Yes vs. No moments
│   └── tool_life_overall.csv      # Overall target moments
│
└── ToolLife_ML_Project/           # Original project workspace & training script
    └── ToolLife_ML_Project/
        └── analyze_tool_life.py   # Training and analysis pipeline script
```

---

## ⚡ Key Features

### 1. Interactive Prediction Interface
- **Inputs:** Lathe speed ($X_1$, 400–1200 rpm), Tool type ($X_2$, A or B), Feed rate ($X_3$, 0.05–0.50 mm/rev), Depth of cut ($X_4$, 0.2–3.0 mm), Coolant usage ($X_5$, Yes/No), and Workpiece hardness ($X_6$, 130–270 HB).
- **Physical Domain Validation:** Warns operators when inputs extrapolate beyond empirical bounds ($[500, 1000]$ rpm, etc.).
- **Live Inferences:** Immediate calculation of tool life in hours, minutes, and standard 8-hour machining shifts.
- **Dynamic Speed Sensitivity Curve:** Real-time Plotly curve tracking tool life across spindle speeds while holding other parameters fixed.
- **Instantaneous What-If Deltas:** Shows the exact gain/loss if toggling tool grade (A $\leftrightarrow$ B) or coolant (Yes $\leftrightarrow$ No).

### 2. Analytics & Metrology Dashboard
- **Summary Statistics:** Central moments (mean, median, standard deviation, IQR, min, max) for all numeric parameters.
- **Missing Value Breakdown:** Documents the 5% missingness pattern across predictors and the 73 complete cases.
- **Pearson Correlations:** Full correlation matrix and target association ranking ($X_1$ is strongest at $r = -0.5037$).
- **Tool-Type & Coolant Comparisons:** Visual boxplots, overlaid histograms, and statistical tests comparing carbide grades and flood coolant impacts.

### 3. Parametric Regression Analysis (OLS)
- **Full Model:** $Y \sim X_1 + X_3 + X_4 + X_6 + \text{tool\_B} + \text{coolant\_Yes}$ ($R^2 = 0.9527$, $\text{Adj } R^2 = 0.9484$, $F = 221.6$).
- **Reduced Model:** $Y \sim X_1 + \text{tool\_B}$ ($R^2 = 0.9269$, $\text{Adj } R^2 = 0.9248$, $F = 443.5$).
- **Statistical Interpretation:** Marginal effect sizes, 95% confidence intervals, and hypothesis tests ($p$-values).
- **Diagnostic Transparency:** Condition number ($1.34 \times 10^4$) and Durbin-Watson ($1.894$) explanations.

### 4. Model Evaluation & Benchmark
- Actual saved cross-validation and test set performance:
  | Model | 5-Fold CV MAE (hrs) | Test MAE (hrs) | Test RMSE (hrs) | Test $R^2$ |
  | :--- | :---: | :---: | :---: | :---: |
  | **Ridge Regression (Selected)** | **2.2623** | **2.6399** | **4.3181** | **0.6928** |
  | Extra Trees Regressor | 2.8071 | 3.1714 | 4.9955 | 0.5888 |
  | Gradient Boosting Regressor | 2.8566 | 3.3773 | 5.3084 | 0.5357 |
  | Random Forest Regressor | 3.0565 | 3.4845 | 5.3270 | 0.5324 |
- **Engineering Justification:** Explanation of why L2-regularized linear models outperform tree ensembles on small-sample engineering datasets ($N=100$).

### 5. Original Assignment Solutions
- Comprehensive cards displaying all five questions and exact calculated solutions from the original workbook.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 (64-bit)

### Installation
Clone or navigate to the repository directory:
```bash
cd "d:/COLLEGE/GitHub Projects/ML EXAM/TOOL LIFE"
```

Create and activate a virtual environment (optional but recommended):
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the application directly using:
```bash
streamlit run app.py
```

If Streamlit is installed within a specific virtual environment:
```bash
.\ToolLife_ML_Project\ToolLife_ML_Project\venv\Scripts\streamlit.exe run app.py
```

The application will launch on `http://localhost:8501`.

---

## 📊 Dataset & Variables

| Column | Variable Name | Physical Meaning | Typical Range / Domain | Missing Count |
| :---: | :--- | :--- | :--- | :---: |
| **Y** | Tool Life | Operational cutting hours until flank wear threshold | 9.10 – 43.67 hrs | 0 (0.0%) |
| **X1** | Lathe Speed | Spindle rotational velocity | 500 – 1000 rpm | 5 (5.0%) |
| **X2** | Tool Type | Carbide tool metallurgical grade (A vs. B) | A (48 runs), B (47 runs) | 5 (5.0%) |
| **X3** | Feed Rate | Axis travel distance per workpiece revolution | 0.10 – 0.40 mm/rev | 5 (5.0%) |
| **X4** | Depth of Cut | Radial tool engagement depth into workpiece | 0.50 – 2.50 mm | 5 (5.0%) |
| **X5** | Coolant | Flood coolant application | Yes (57 runs), No (38 runs) | 5 (5.0%) |
| **X6** | Workpiece Hardness | Metallurgical material hardness | 150 – 249 Brinell (HB) | 5 (5.0%) |

---

## 📐 The Five Assignment Solutions Summary

1. **Question 1 (Missingness):** 
   Each predictor ($X_1$ to $X_6$) has exactly 5 missing values (5.0%); target $Y$ has 0 (0.0%). There are 73 complete cases.
2. **Question 2 (Descriptive Moments):** 
   - Overall $Y$: Mean = 24.22h, Median = 24.26h, SD = 9.25h.
   - Tool A: Mean = 16.52h, Median = 15.33h, SD = 4.80h.
   - Tool B: Mean = 32.07h, Median = 32.58h, SD = 5.73h.
   - Coolant No: Mean = 22.92h, Median = 21.97h, SD = 8.66h.
   - Coolant Yes: Mean = 25.34h, Median = 25.40h, SD = 9.62h.
3. **Question 3 (Correlations):** 
   $X_1$ ($r = -0.5037$, strongest negative correlation), $X_4$ ($r = -0.1252$), $X_3$ ($r = -0.1081$), $X_6$ ($r = -0.0339$).
4. **Question 4 (OLS Regression):** 
   Full Model $R^2 = 0.9527$ ($\text{Adj } R^2 = 0.9484$). Reduced Model $R^2 = 0.9269$ ($\text{Adj } R^2 = 0.9248$). Adding the remaining variables improves $R^2$ by 0.0258 (+2.58%).
5. **Question 5 (Coolant Difference):** 
   Unadjusted mean with coolant = 25.3386h; without coolant = 22.9189h. Bivariate difference = +2.4196 hours.

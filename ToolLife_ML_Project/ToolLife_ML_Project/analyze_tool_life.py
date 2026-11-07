"""
ToolLife ML: end-to-end tool life analysis and prediction.
Run:
    pip install -r requirements.txt
    python analyze_tool_life.py --input TOOL_LIFE_DATA_100.xlsx

Outputs are written to ./outputs.
The OLS answers use complete cases to make the requested coefficients reproducible.
The predictive ML pipeline imputes missing predictors inside each training fold to avoid leakage.
"""
import argparse, json, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6"]
NUMERIC = ["X1", "X3", "X4", "X6"]
CATEGORICAL = ["X2", "X5"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="TOOL_LIFE_DATA_100.xlsx")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(args.input, sheet_name="Data")
    required = ["Y"] + FEATURES
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    data = raw[required].copy()
    for c in ["Y"] + NUMERIC:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    for c in CATEGORICAL:
        # Convert pandas nullable strings to ordinary Python strings/np.nan;
        # this keeps SimpleImputer compatible across scikit-learn versions.
        data[c] = data[c].apply(lambda v: str(v).strip() if pd.notna(v) else np.nan)
        data[c] = data[c].replace({"": np.nan, "nan": np.nan})

    # Q1: missingness (Y is expected to have no missing values)
    missing = pd.DataFrame({
        "column": data.columns,
        "missing_count": data.isna().sum().values,
        "missing_percent": (data.isna().mean().values * 100).round(2)
    })
    missing.to_csv(out / "missing_values.csv", index=False)

    # Q2: descriptive statistics
    overall = data["Y"].agg(["count", "mean", "median", "std"]).to_frame("overall")
    by_tool = data.groupby("X2", dropna=False)["Y"].agg(["count","mean","median","std"])
    by_coolant = data.groupby("X5", dropna=False)["Y"].agg(["count","mean","median","std"])
    overall.to_csv(out / "tool_life_overall.csv")
    by_tool.to_csv(out / "tool_life_by_tool_type.csv")
    by_coolant.to_csv(out / "tool_life_by_coolant.csv")

    # Q3: Pearson correlations (pairwise complete observations)
    correlations = data[["Y"] + NUMERIC].corr()["Y"].drop("Y").sort_values(key=np.abs, ascending=False)
    correlations.rename("pearson_r").to_csv(out / "correlations.csv")

    # Q4: requested OLS models. Complete-case analysis is explicit and reproducible.
    cc = data.dropna(subset=required).copy()
    cc["tool_B"] = (cc["X2"] == "B").astype(int)
    cc["coolant_Yes"] = (cc["X5"] == "Yes").astype(int)
    full = smf.ols("Y ~ X1 + X3 + X4 + X6 + tool_B + coolant_Yes", data=cc).fit()
    reduced = smf.ols("Y ~ X1 + tool_B", data=cc).fit()
    (out / "full_ols_summary.txt").write_text(full.summary().as_text(), encoding="utf-8")
    (out / "reduced_ols_summary.txt").write_text(reduced.summary().as_text(), encoding="utf-8")
    coef = pd.DataFrame({"full_model": full.params})
    coef.to_csv(out / "full_ols_coefficients.csv")
    pd.DataFrame({"reduced_model": reduced.params}).to_csv(out / "reduced_ols_coefficients.csv")

    # Q5: unadjusted mean difference; this is association, not proof of causation.
    coolant_means = data.groupby("X5")["Y"].agg(["count","mean","median","std"])
    coolant_means.to_csv(out / "coolant_comparison.csv")
    coolant_difference = float(coolant_means.loc["Yes", "mean"] - coolant_means.loc["No", "mean"])

    # Visualisations
    plt.figure(figsize=(7,5))
    plt.hist(data["Y"].dropna(), bins=15, edgecolor="black")
    plt.xlabel("Tool life (hours)"); plt.ylabel("Frequency"); plt.title("Distribution of tool life")
    plt.tight_layout(); plt.savefig(out / "tool_life_distribution.png", dpi=160); plt.close()

    plt.figure(figsize=(7,5))
    plt.scatter(data["X1"], data["Y"], alpha=.8)
    plt.xlabel("Lathe speed (rpm)"); plt.ylabel("Tool life (hours)")
    plt.title("Tool life vs lathe speed")
    plt.tight_layout(); plt.savefig(out / "tool_life_vs_speed.png", dpi=160); plt.close()

    # ML: hold-out test set; all preprocessing lives inside the Pipeline.
    X, y = data[FEATURES], data["Y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])
    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=1.0, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=1.0, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=2,
            min_samples_leaf=3, loss="huber", random_state=42
        )
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        cv_mae = -cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
        ).mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "model": name, "train_5fold_cv_MAE": cv_mae,
            "test_MAE": mean_absolute_error(y_test, pred),
            "test_RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "test_R2": r2_score(y_test, pred)
        })
        fitted[name] = pipe
    results = pd.DataFrame(rows).sort_values("train_5fold_cv_MAE")
    results.to_csv(out / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["model"]

    # Final production model: refit selected model on all available labelled rows.
    final_model = fitted[best_name]
    final_model.fit(X, y)
    joblib.dump(final_model, out / "tool_life_model.joblib")

    # Write a concise text summary and a machine-readable JSON.
    summary = {
        "rows": int(len(data)),
        "complete_cases_for_ols": int(len(cc)),
        "missing_values": missing.set_index("column")["missing_count"].to_dict(),
        "overall_tool_life_mean": float(data["Y"].mean()),
        "overall_tool_life_median": float(data["Y"].median()),
        "overall_tool_life_std": float(data["Y"].std()),
        "strongest_numeric_correlation": str(correlations.index[0]),
        "strongest_numeric_correlation_value": float(correlations.iloc[0]),
        "full_ols_r2": float(full.rsquared),
        "reduced_ols_r2": float(reduced.rsquared),
        "coolant_yes_minus_no_mean_hours": coolant_difference,
        "selected_ml_model_by_training_cv_mae": best_name
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\\nModel comparison:")
    print(results.to_string(index=False))
    print(f"\\nOutputs saved to: {out.resolve()}")

if __name__ == "__main__":
    main()

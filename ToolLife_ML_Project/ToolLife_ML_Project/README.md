# ToolLife ML — Tool Life Prediction and Regression Analysis

## Project objective
Predict machining tool life (Y, hours) from lathe speed, tool type, feed rate,
depth of cut, coolant usage, and workpiece hardness. The project also answers
the five questions in the supplied Excel workbook.

## Dataset
- 100 observations
- Target: `Y`
- Features: `X1`, `X2`, `X3`, `X4`, `X5`, `X6`
- Missing predictor values are handled inside a scikit-learn pipeline:
  median imputation for numeric features and most-frequent imputation for
  categorical features. Categorical values are one-hot encoded.

## Run locally
1. Install Python 3.10 or newer.
2. Put `TOOL_LIFE_DATA_100.xlsx` in this folder.
3. Create/activate a virtual environment (recommended).
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python analyze_tool_life.py --input TOOL_LIFE_DATA_100.xlsx`

The script creates an `outputs/` folder with the descriptive statistics,
correlations, OLS summaries, model comparison, plots, JSON summary, and the
trained `tool_life_model.joblib` file.

## Methodology
- The five assignment questions are answered using descriptive statistics,
  Pearson correlation, and ordinary least squares (OLS).
- The requested OLS models use complete cases so the coefficients and R²
  correspond to a clearly stated sample (rows with all required fields).
- Prediction models use a train/test split (80/20), 5-fold CV on the training
  portion for model selection, and preprocessing inside each pipeline to
  prevent data leakage.
- The selected model is refitted on all labelled observations for deployment.

## Important interpretation
This dataset has only 100 rows. Results should be treated as a learning/demo
model, not a production-grade manufacturing quality guarantee. Correlation
and group mean differences do not establish causation. A larger independently
collected dataset and external validation are needed before operational use.

## Example prediction
After training, load the saved model:
```python
import joblib
import pandas as pd

model = joblib.load("outputs/tool_life_model.joblib")
new_run = pd.DataFrame([{
    "X1": 650, "X2": "B", "X3": 0.22,
    "X4": 1.2, "X5": "Yes", "X6": 205
}])
print("Predicted tool life (hours):", model.predict(new_run)[0])
```

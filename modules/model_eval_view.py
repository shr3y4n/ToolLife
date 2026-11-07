"""
model_eval_view.py - Machine Learning Model Benchmarking & Architecture Evaluation.
Compares Ridge, Random Forest, Extra Trees, and Gradient Boosting based on actual saved metrics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.ui_theme import INDUSTRIAL_COLORS, get_industrial_plotly_layout

def render_model_eval_view(model_comp_df: pd.DataFrame):
    """Renders the comprehensive model comparison and evaluation dashboard."""
    
    st.markdown("""
    <div class="section-banner">
        <div>
            <h3>Predictive Machine Learning Benchmark & Validation</h3>
            <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">
                Cross-validated benchmark comparing linear regularisation against non-linear tree ensembles using an 80/20 train/test split.
            </div>
        </div>
        <span>SELECTION CRITERION: 5-FOLD CV MAE ON TRAINING PARTITION</span>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics for Best Model (Ridge)
    best_row = model_comp_df.iloc[0]
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #1d4ed8;">
            <div class="title">Selected Champion Model</div>
            <div class="value" style="font-size: 1.4rem; color: #1d4ed8;">{best_row['model']}</div>
            <div class="subtext">L2 Regularised Linear Regression</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Train 5-Fold CV MAE</div>
            <div class="value">{best_row['train_5fold_cv_MAE']:.3f}<span style="font-size: 1rem;">h</span></div>
            <div class="subtext">Validation error on training fold</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Test Set MAE (20% Holdout)</div>
            <div class="value">{best_row['test_MAE']:.3f}<span style="font-size: 1rem;">h</span></div>
            <div class="subtext">Unseen hold-out accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Test Set R-Squared</div>
            <div class="value" style="color: #047857;">{best_row['test_R2']:.3f}</div>
            <div class="subtext">Test RMSE: {best_row['test_RMSE']:.3f}h</div>
        </div>
        """, unsafe_allow_html=True)

    # Formal Comparison Table
    st.markdown("##### 1. Actual Model Benchmark Results (Saved Outputs)")
    
    disp_df = model_comp_df.copy()
    disp_df.columns = ["Model Architecture", "Train 5-Fold CV MAE (hrs)", "Test MAE (hrs)", "Test RMSE (hrs)", "Test R²"]
    
    # Format and highlight best
    st.dataframe(
        disp_df.style.format({
            "Train 5-Fold CV MAE (hrs)": "{:.4f}",
            "Test MAE (hrs)": "{:.4f}",
            "Test RMSE (hrs)": "{:.4f}",
            "Test R²": "{:.4f}"
        }).highlight_min(subset=["Train 5-Fold CV MAE (hrs)", "Test MAE (hrs)", "Test RMSE (hrs)"], color="#dbeafe")
          .highlight_max(subset=["Test R²"], color="#dbeafe"),
        use_container_width=True,
        hide_index=True
    )
    st.caption("Lower is better for MAE/RMSE. Higher is better for R². Ridge achieved optimal scores across all benchmark dimensions.")

    # Visual Charts
    st.markdown("##### 2. Performance Comparison Across Algorithms")
    c_chart1, c_chart2 = st.columns(2, gap="large")

    with c_chart1:
        # Cross-validation MAE & Test MAE
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Bar(
            name="Train 5-Fold CV MAE",
            x=model_comp_df["model"],
            y=model_comp_df["train_5fold_cv_MAE"],
            marker_color="#1d4ed8",
            text=[f"{v:.3f}h" for v in model_comp_df["train_5fold_cv_MAE"]],
            textposition="auto"
        ))
        fig_mae.add_trace(go.Bar(
            name="Holdout Test MAE",
            x=model_comp_df["model"],
            y=model_comp_df["test_MAE"],
            marker_color="#60a5fa",
            text=[f"{v:.3f}h" for v in model_comp_df["test_MAE"]],
            textposition="auto"
        ))
        layout_mae = get_industrial_plotly_layout(
            height=340,
            title="Mean Absolute Error (CV MAE vs. Test MAE)",
            xlab="Model Candidate",
            ylab="MAE [hours]"
        )
        fig_mae.update_layout(layout_mae)
        fig_mae.update_layout(barmode="group")
        st.plotly_chart(fig_mae, use_container_width=True)

    with c_chart2:
        # Test Set R2 Comparison
        fig_r2 = go.Figure(data=[
            go.Bar(
                x=model_comp_df["model"],
                y=model_comp_df["test_R2"],
                marker=dict(color=["#047857" if m == "Ridge" else "#64748b" for m in model_comp_df["model"]]),
                text=[f"{v:.3f}" for v in model_comp_df["test_R2"]],
                textposition="auto"
            )
        ])
        layout_r2 = get_industrial_plotly_layout(
            height=340,
            title="Holdout Test R-Squared Comparison",
            xlab="Model Candidate",
            ylab="R² Score (Holdout)"
        )
        fig_r2.update_layout(layout_r2)
        fig_r2.update_yaxes(range=[0, 0.8])
        st.plotly_chart(fig_r2, use_container_width=True)

    # Engineering Analysis: Why Linear Regularization Triumphed
    st.markdown("##### 3. Engineering & Statistical Rationale: Why Ridge Outperformed Tree Ensembles")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown("""
        **1. Sample Regime (N=100)**
        - Ensembles like Random Forest and Extra Trees require hundreds or thousands of observations to construct stable orthogonal decision boundaries.
        - With only 80 training samples, high-depth tree partitions split sparse data into noisy terminal leaves.
        """)
    with col_r2:
        st.markdown("""
        **2. Underlying Physical Topology**
        - Tool wear follows Taylor's continuous thermal law ($V \cdot T^n = C$).
        - The actual physical relationship is smooth and monotonic.
        - Ridge fits smooth hyperplanes naturally without the step-function artifacts of decision trees.
        """)
    with col_r3:
        st.markdown("""
        **3. L2 Weight Shrinkage**
        - Ridge penalizes large regression weights through $\lambda \sum w_j^2$.
        - This effectively suppresses noise from collinear or weakly correlated inputs (hardness $X_6$, depth $X_4$), preserving test set generalizability.
        """)

    # SECTION 4: PREPROCESSING & PIPELINE ARCHITECTURE
    st.markdown("##### 4. Production Scikit-Learn Pipeline Architecture")
    st.markdown("""
    To eliminate data leakage, all imputation and scaling occur strictly inside the pipeline folds:
    """)
    
    st.markdown("""
    ```
    Raw Input (X1, X2, X3, X4, X5, X6)
       │
       ├─── Numeric Sub-Pipeline (X1, X3, X4, X6)
       │       ├── SimpleImputer(strategy='median')
       │       └── StandardScaler()
       │
       ├─── Categorical Sub-Pipeline (X2, X5)
       │       ├── SimpleImputer(strategy='most_frequent')
       │       └── OneHotEncoder(handle_unknown='ignore')
       │
       ▼
    Combined Preprocessed Feature Vector
       │
       ▼
    Ridge Regression Estimator (alpha=1.0)
       │
       ▼
    Final Deployment Model: Refitted on All N=100 Observations
    ```
    """)
    
    st.info("The production model saved at `outputs/tool_life_model.joblib` was refitted on the entire 100-sample dataset after model selection, ensuring maximum operational fidelity for live shop-floor deployment.")

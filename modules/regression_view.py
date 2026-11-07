"""
regression_view.py - Econometric & Ordinary Least Squares (OLS) Regression Analysis.
Presents the complete-case full and reduced regression equations, coefficients,
hypothesis tests (p-values, CIs), R-squared metrics, and diagnostic explanations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.ui_theme import INDUSTRIAL_COLORS, get_industrial_plotly_layout

# Verified regression statistics from outputs/
FULL_MODEL_PARAMS = [
    {"variable": "Intercept", "coef": 49.9403, "std_err": 3.243, "t": 15.400, "p_val": "< 0.001", "ci_lower": 43.466, "ci_upper": 56.415, "sig": "***"},
    {"variable": "X1 (Lathe Speed)", "coef": -0.0302, "std_err": 0.002, "t": -17.466, "p_val": "< 0.001", "ci_lower": -0.034, "ci_upper": -0.027, "sig": "***"},
    {"variable": "X3 (Feed Rate)", "coef": -13.2674, "std_err": 4.296, "t": -3.089, "p_val": "0.003", "ci_lower": -21.844, "ci_upper": -4.691, "sig": "**"},
    {"variable": "X4 (Depth of Cut)", "coef": -1.7037, "std_err": 0.595, "t": -2.863, "p_val": "0.006", "ci_lower": -2.892, "ci_upper": -0.516, "sig": "**"},
    {"variable": "X6 (Workpiece Hardness)", "coef": -0.0266, "std_err": 0.014, "t": -1.883, "p_val": "0.064", "ci_lower": -0.055, "ci_upper": 0.002, "sig": "ns"},
    {"variable": "tool_B (Tool Type = B)", "coef": 15.4615, "std_err": 0.506, "t": 30.527, "p_val": "< 0.001", "ci_lower": 14.450, "ci_upper": 16.473, "sig": "***"},
    {"variable": "coolant_Yes (Coolant = Yes)", "coef": 0.6008, "std_err": 0.542, "t": 1.109, "p_val": "0.271", "ci_lower": -0.481, "ci_upper": 1.682, "sig": "ns"}
]

REDUCED_MODEL_PARAMS = [
    {"variable": "Intercept", "coef": 39.7234, "std_err": 1.577, "t": 25.195, "p_val": "< 0.001", "ci_lower": 36.579, "ci_upper": 42.868, "sig": "***"},
    {"variable": "X1 (Lathe Speed)", "coef": -0.0310, "std_err": 0.002, "t": -15.163, "p_val": "< 0.001", "ci_lower": -0.035, "ci_upper": -0.027, "sig": "***"},
    {"variable": "tool_B (Tool Type = B)", "coef": 15.2081, "std_err": 0.607, "t": 25.045, "p_val": "< 0.001", "ci_lower": 13.997, "ci_upper": 16.419, "sig": "***"}
]

def render_regression_view(full_text: str, reduced_text: str):
    """Renders the OLS econometric analysis section."""
    
    st.markdown("""
    <div class="section-banner">
        <div>
            <h3>Ordinary Least Squares (OLS) Regression Analysis</h3>
            <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">
                Complete-case parametric modeling (n=73) comparing full feature specification against reduced two-factor model.
            </div>
        </div>
        <span>METHODOLOGY: STATSMODELS OLS (EXACT PARAMETRIC)</span>
    </div>
    """, unsafe_allow_html=True)

    # High-level Comparison Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="title">Full Model R²</div>
            <div class="value">0.9527</div>
            <div class="subtext">Adj R² = 0.9484 (df=66)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="title">Reduced Model R²</div>
            <div class="value">0.9269</div>
            <div class="subtext">Adj R² = 0.9248 (df=70)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="title">Variance Delta (ΔR²)</div>
            <div class="value" style="color: #047857;">+0.0258</div>
            <div class="subtext">+2.58% explained variance</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="title">Primary Driver</div>
            <div class="value" style="font-size: 1.25rem;">X1 + Tool B</div>
            <div class="subtext">Accounts for 92.69% of variance</div>
        </div>
        """, unsafe_allow_html=True)

    # Comparative Summary Table
    st.markdown("##### 1. Model Architecture & Goodness-of-Fit Comparison")
    comp_df = pd.DataFrame([
        {
            "Model Specification": "Full OLS Model",
            "Formula / Regressors": "Y ~ X1 + X3 + X4 + X6 + tool_B + coolant_Yes",
            "Obs (n)": 73,
            "R-squared": "0.9527",
            "Adj. R-squared": "0.9484",
            "F-Statistic": "221.6 (p=1.01e-41)",
            "Log-Likelihood": "-155.61",
            "AIC": "325.2",
            "BIC": "341.3"
        },
        {
            "Model Specification": "Reduced OLS Model",
            "Formula / Regressors": "Y ~ X1 + tool_B",
            "Obs (n)": 73,
            "R-squared": "0.9269",
            "Adj. R-squared": "0.9248",
            "F-Statistic": "443.5 (p=1.77e-40)",
            "Log-Likelihood": "-171.52",
            "AIC": "349.0",
            "BIC": "355.9"
        }
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="tech-box">
        <strong>Key Engineering Takeaway:</strong> 
        The reduced model incorporating solely <strong>Lathe Speed (X1)</strong> and <strong>Tool Type (tool_B)</strong> captures an astounding <strong>92.69% of all variance</strong> in tool life.
        Adding feed rate, depth of cut, hardness, and coolant into the full model increases R² by <strong>+0.0258</strong> (+2.58 percentage points) and lowers AIC from 349.0 to 325.2, confirming statistical superiority for precise parameter tuning.
    </div>
    """, unsafe_allow_html=True)

    # SECTION 2: FULL OLS DETAILED COEFFICIENTS
    st.markdown("##### 2. Full OLS Model: Detailed Parameter Estimates")
    st.latex(r"Y = \beta_0 + \beta_1 X_1 + \beta_3 X_3 + \beta_4 X_4 + \beta_6 X_6 + \beta_{\text{tool\_B}} \cdot \text{tool\_B} + \beta_{\text{coolant}} \cdot \text{coolant\_Yes} + \varepsilon")
    
    full_df = pd.DataFrame(FULL_MODEL_PARAMS)
    full_table = full_df[["variable", "coef", "std_err", "t", "p_val", "ci_lower", "ci_upper", "sig"]].copy()
    full_table.columns = ["Regressor", "Coefficient (β)", "Std Error", "t-statistic", "p-value", "95% CI Lower", "95% CI Upper", "Significance"]
    st.dataframe(full_table, use_container_width=True, hide_index=True)
    st.caption("Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant (alpha = 0.05).")

    # SECTION 3: COEFFICIENT VISUALISATION (FOREST PLOT)
    st.markdown("##### 3. Standardized Effect Sizes & Confidence Intervals")
    
    fig_ci = go.Figure()
    plot_vars = [p for p in FULL_MODEL_PARAMS if p["variable"] != "Intercept"]
    
    y_names = [p["variable"] for p in plot_vars]
    coefs = [p["coef"] for p in plot_vars]
    err_low = [p["coef"] - p["ci_lower"] for p in plot_vars]
    err_high = [p["ci_upper"] - p["coef"] for p in plot_vars]
    colors = ["#047857" if p["sig"] != "ns" else "#94a3b8" for p in plot_vars]

    fig_ci.add_trace(go.Scatter(
        x=coefs,
        y=y_names,
        mode="markers",
        marker=dict(size=9, color=colors),
        error_x=dict(
            type="data",
            symmetric=False,
            array=err_high,
            arrayminus=err_low,
            color="#475569",
            thickness=2,
            width=6
        ),
        hovertemplate="<b>%{y}</b><br>Coef: %{x:.4f}<extra></extra>"
    ))

    fig_ci.add_vline(x=0, line_dash="dash", line_color="#cbd5e1", line_width=1.5)

    layout_ci = get_industrial_plotly_layout(
        height=340,
        title="Full OLS Model: Estimated Regression Coefficients with 95% Confidence Intervals",
        xlab="Estimated Marginal Effect on Tool Life (Hours)",
        ylab="Model Variable"
    )
    fig_ci.update_layout(layout_ci)
    st.plotly_chart(fig_ci, use_container_width=True)

    # SECTION 4: PHYSICAL INTERPRETATION OF COEFFICIENTS
    st.markdown("##### 4. Physical Metrology & Metallurgical Interpretation")
    col_int1, col_int2 = st.columns(2, gap="large")
    
    with col_int1:
        st.markdown("""
        **1. Lathe Speed (X1) [β = -0.0302, p < 0.001]:**
        - Every **100 rpm increase** in spindle velocity reduces tool life by **3.02 hours**.
        - Dominates tool degradation due to Taylor flank wear physics and thermal softening at the tool-chip interface.
        
        **2. Tool Type B Dummy [β = +15.4615, p < 0.001]:**
        - Switching from standard Tool A to premium Tool B grants an instantaneous **+15.46 hours** life advantage when all other cutting conditions are held identical.
        - By far the single largest technological lever available to process engineers.
        
        **3. Feed Rate (X3) [β = -13.2674, p = 0.003]:**
        - Increasing feed rate by 0.10 mm/rev reduces tool life by **~1.33 hours**.
        - Higher chip load increases shear friction and mechanical force on the cutting edge.
        """)

    with col_int2:
        st.markdown("""
        **4. Depth of Cut (X4) [β = -1.7037, p = 0.006]:**
        - Each 1.0 mm increase in depth of cut shortens tool life by **1.70 hours**.
        - Broadens the tool engagement width, spreading heat along the primary cutting edge.
        
        **5. Workpiece Hardness (X6) [β = -0.0266, p = 0.064]:**
        - A 10 HB increase correlates with -0.27 hours shorter tool life.
        - Marginally non-significant at the 5% level (p=0.064), suggesting material variations in the 150–250 HB range have modest impact relative to speed and tool metallurgy.
        
        **6. Coolant (coolant_Yes) [β = +0.6008, p = 0.271]:**
        - After controlling for speed and tool grade, coolant contributes **+0.60 hours** (36 minutes), which is **not statistically significant** (p=0.271).
        """)

    # SECTION 5: REDUCED MODEL SUMMARY & DIAGNOSTICS
    st.markdown("##### 5. Reduced OLS Model: Detailed Parameter Estimates")
    st.latex(r"Y = 39.7234 - 0.0310 \cdot X_1 + 15.2081 \cdot \text{tool\_B}")
    red_df = pd.DataFrame(REDUCED_MODEL_PARAMS)
    red_table = red_df[["variable", "coef", "std_err", "t", "p_val", "ci_lower", "ci_upper", "sig"]].copy()
    red_table.columns = ["Regressor", "Coefficient (β)", "Std Error", "t-statistic", "p-value", "95% CI Lower", "95% CI Upper", "Significance"]
    st.dataframe(red_table, use_container_width=True, hide_index=True)

    # Collapsible Original Statsmodels Summaries
    with st.expander("📄 View Full Statsmodels Summary Output (outputs/full_ols_summary.txt)"):
        st.code(full_text if full_text else "Full OLS summary text file not found.", language="text")

    with st.expander("📄 View Reduced Statsmodels Summary Output (outputs/reduced_ols_summary.txt)"):
        st.code(reduced_text if reduced_text else "Reduced OLS summary text file not found.", language="text")

    # Condition Number and Multicollinearity Note
    st.markdown("""
    <div class="tech-box warning">
        <strong>Condition Number & Multicollinearity Diagnostic:</strong><br>
        The full regression reports a Condition Number of <code>1.34e+04</code>. 
        In mechanical engineering datasets, this occurs naturally when unstandardized features have wildly different physical units 
        (e.g., Lathe speed ~1000 rpm vs. Feed rate ~0.2 mm/rev). 
        The Durbin-Watson statistic is <code>1.894</code> (near 2.0), confirming zero first-order autocorrelation in residuals.
    </div>
    """, unsafe_allow_html=True)

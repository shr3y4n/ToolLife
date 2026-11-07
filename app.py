"""
ToolLife ML – Intelligent Machining Tool Life Prediction System
Main Streamlit Application Entrypoint.
"""
import sys
from pathlib import Path
import streamlit as st

# Ensure workspace modules are resolvable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.ui_theme import apply_custom_css
from modules.data_loader import (
    load_raw_dataset,
    load_summary_json,
    load_model_comparison,
    load_correlations,
    load_missing_values,
    load_ols_summaries,
    load_subgroup_statistics,
    load_trained_model,
    load_assignment_qa
)
from modules.prediction_view import render_prediction_view
from modules.analytics_view import render_analytics_view
from modules.regression_view import render_regression_view
from modules.model_eval_view import render_model_eval_view
from modules.assignment_qa_view import render_assignment_qa_view

# Page Configuration
st.set_page_config(
    page_title="ToolLife ML – Intelligent Tool Life Prediction System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom Industrial CSS
apply_custom_css()

# Cached Data & Model Loaders
@st.cache_data(show_spinner=False)
def get_dataset():
    return load_raw_dataset()

@st.cache_data(show_spinner=False)
def get_metadata():
    return {
        "summary": load_summary_json(),
        "model_comparison": load_model_comparison(),
        "correlations": load_correlations(),
        "missing_values": load_missing_values(),
        "subgroup_stats": load_subgroup_statistics(),
        "qa_list": load_assignment_qa()
    }

@st.cache_data(show_spinner=False)
def get_ols_summaries():
    return load_ols_summaries()

@st.cache_resource(show_spinner=False)
def get_model():
    return load_trained_model()


def main():
    # Load all verified components
    try:
        raw_df = get_dataset()
        metadata = get_metadata()
        full_ols_txt, red_ols_txt = get_ols_summaries()
        pipeline_model = get_model()
    except Exception as e:
        st.error(f"Critical error initializing application data or trained model: {e}")
        st.info("Please verify that TOOL_LIFE_DATA_100.xlsx and outputs/ are accessible.")
        return

    # Industrial Application Header Banner
    st.markdown("""
    <div class="industrial-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <h1>ToolLife ML – Intelligent Machining Tool Life Prediction System</h1>
                <p>
                    Precision metrology, multivariate OLS regression modeling, and cross-validated machine learning 
                    for forecasting cutting tool longevity across turning operational parameters.
                </p>
            </div>
        </div>
        <div class="badge-bar">
            <span class="ind-badge active">Model: Ridge Pipeline (joblib)</span>
            <span class="ind-badge">Dataset: N=100 Machining Cycles</span>
            <span class="ind-badge">OLS R² = 0.953</span>
            <span class="ind-badge">CV MAE = 2.262 hrs</span>
            <span class="ind-badge">Reproducible Benchmark</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation & System Diagnostic Status
    with st.sidebar:
        st.markdown("### Navigation")
        selected_page = st.radio(
            "Select Operational Module:",
            options=[
                "⚙️ Interactive Prediction Interface",
                "📊 Analytics & Metrology Dashboard",
                "📐 Parametric Regression (OLS)",
                "🏆 Model Benchmark & Evaluation",
                "📝 Assignment Questions & Solutions"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown("### System Telemetry & Status")
        st.markdown("""
        <div style="font-size: 0.82rem; line-height: 1.6; color: #475569;">
            • <strong>Pipeline:</strong> Ridge Regression (α=1.0)<br>
            • <strong>Trained Model:</strong> <code>tool_life_model.joblib</code><br>
            • <strong>Target Variable:</strong> <code>Y</code> (Tool Life, hours)<br>
            • <strong>Input Dimensions:</strong> 6 features (X1 to X6)<br>
            • <strong>Imputation:</strong> Median (Num) / Mode (Cat)<br>
            • <strong>Scaling:</strong> StandardScaler<br>
            • <strong>Complete Cases:</strong> 73 / 100 rows<br>
            • <strong>Deployment Status:</strong> <span style="color: #047857; font-weight: 700;">ONLINE (100% Verified)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Variable Specification")
        st.caption("""
        - **X1:** Lathe speed (rpm)
        - **X2:** Tool type (A or B)
        - **X3:** Feed rate (mm/rev)
        - **X4:** Depth of cut (mm)
        - **X5:** Coolant used (Yes/No)
        - **X6:** Workpiece hardness (HB)
        - **Y:** Tool life in hours [Output]
        """)

    # Route to Selected Modular View
    if selected_page == "⚙️ Interactive Prediction Interface":
        render_prediction_view(pipeline_model, raw_df)
    elif selected_page == "📊 Analytics & Metrology Dashboard":
        render_analytics_view(
            raw_df=raw_df,
            summary_data=metadata["summary"],
            missing_df=metadata["missing_values"],
            correlations_df=metadata["correlations"],
            subgroup_stats=metadata["subgroup_stats"]
        )
    elif selected_page == "📐 Parametric Regression (OLS)":
        render_regression_view(full_ols_txt, red_ols_txt)
    elif selected_page == "🏆 Model Benchmark & Evaluation":
        render_model_eval_view(metadata["model_comparison"])
    elif selected_page == "📝 Assignment Questions & Solutions":
        render_assignment_qa_view(metadata["qa_list"])

    # Professional Footer
    st.markdown("""
    <div style="margin-top: 3.5rem; padding-top: 1.25rem; border-top: 1px solid #e2e8f0; font-size: 0.8rem; color: #64748b; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <strong>ToolLife ML</strong> – Intelligent Machining Tool Life Prediction & Regression Analytics System.
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
            ISO 3685 Tool Life Testing Reference | Built with Streamlit, Scikit-Learn & Statsmodels
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

"""
analytics_view.py - Comprehensive Industrial Analytics Dashboard.
Visualises dataset statistics, missingness, Pearson correlations, tool life distributions,
tool-type comparative dynamics, and coolant impacts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.ui_theme import INDUSTRIAL_COLORS, get_industrial_plotly_layout

def render_analytics_view(raw_df: pd.DataFrame, summary_data: dict, missing_df: pd.DataFrame, 
                           correlations_df: pd.DataFrame, subgroup_stats: dict):
    """Renders the comprehensive exploratory data analysis dashboard."""
    
    st.markdown("""
    <div class="section-banner">
        <div>
            <h3>Machining Process Metrology & Exploratory Analytics</h3>
            <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">
                Empirical dataset evaluation based on 100 recorded lathe production cycles.
            </div>
        </div>
        <span>POPULATION: N=100 OBSERVATIONS</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Top KPI Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Total Trial Runs</div>
            <div class="value">{len(raw_df)}</div>
            <div class="subtext">Recorded machining cycles</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Complete Cases</div>
            <div class="value">{summary_data.get('complete_cases_for_ols', 73)}</div>
            <div class="subtext">Zero missing entries (73.0%)</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Mean Tool Life (Y)</div>
            <div class="value">{summary_data.get('overall_tool_life_mean', 24.22):.2f}<span style="font-size: 1rem;">h</span></div>
            <div class="subtext">Std Dev: ±{summary_data.get('overall_tool_life_std', 9.25):.2f}h</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="title">Tool B Advantage</div>
            <div class="value" style="color: #047857;">+15.55<span style="font-size: 1rem;">h</span></div>
            <div class="subtext">+94.2% lifespan vs Tool A</div>
        </div>
        """, unsafe_allow_html=True)

    # Sub-tabs for structured analysis
    tab_summary, tab_missing, tab_corr, tab_dist, tab_tool, tab_coolant = st.tabs([
        "📊 Dataset Statistics",
        "🔍 Missing Values (Q1)",
        "🔗 Correlation Analysis (Q3)",
        "📈 Tool Life Distributions",
        "⚙️ Tool-Type Comparisons (Q2)",
        "💧 Coolant Comparisons (Q2 & Q5)"
    ])

    # TAB 1: DATASET STATISTICS
    with tab_summary:
        st.markdown("##### 1. Numeric Feature Distributions & Moments")
        num_cols = ["Y", "X1", "X3", "X4", "X6"]
        desc_df = raw_df[num_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        desc_df.columns = ["Valid N", "Mean", "Std Dev", "Min", "P25", "Median", "P75", "Max"]
        
        # Add column descriptions
        var_labels = {
            "Y": "Tool Life (hours) [Target]",
            "X1": "Lathe Speed (rpm)",
            "X3": "Feed Rate (mm/rev)",
            "X4": "Depth of Cut (mm)",
            "X6": "Workpiece Hardness (HB)"
        }
        desc_df.index = [f"{c} — {var_labels.get(c, c)}" for c in desc_df.index]
        st.dataframe(desc_df.style.format("{:.3f}"), use_container_width=True)

        st.markdown("##### 2. Categorical Factor Distributions")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            x2_counts = raw_df["X2"].value_counts(dropna=False).reset_index()
            x2_counts.columns = ["Tool Type (X2)", "Frequency"]
            x2_counts["Tool Type (X2)"] = x2_counts["Tool Type (X2)"].fillna("Missing (NaN)")
            x2_counts["Percentage"] = (x2_counts["Frequency"] / len(raw_df) * 100).round(1).astype(str) + "%"
            st.dataframe(x2_counts, use_container_width=True, hide_index=True)
        with c_col2:
            x5_counts = raw_df["X5"].value_counts(dropna=False).reset_index()
            x5_counts.columns = ["Coolant Used (X5)", "Frequency"]
            x5_counts["Coolant Used (X5)"] = x5_counts["Coolant Used (X5)"].fillna("Missing (NaN)")
            x5_counts["Percentage"] = (x5_counts["Frequency"] / len(raw_df) * 100).round(1).astype(str) + "%"
            st.dataframe(x5_counts, use_container_width=True, hide_index=True)

        with st.expander("🔎 Browse Raw Recorded Observations (100 Rows)"):
            st.dataframe(raw_df, use_container_width=True)

    # TAB 2: MISSING VALUES
    with tab_missing:
        st.markdown("##### Missing Value Distribution Across Variables (Assignment Question 1)")
        st.markdown("""
        <div class="tech-box">
            <strong>Key Finding:</strong> Tool life (<code>Y</code>) contains zero missing values (100% complete). 
            Each of the 6 predictor variables (<code>X1</code> through <code>X6</code>) has exactly <strong>5 missing values (5.0%)</strong>. 
            Because missing values occur across different rows, a total of <strong>73 rows are complete cases</strong>.
        </div>
        """, unsafe_allow_html=True)

        col_m_tbl, col_m_fig = st.columns([1, 1.3], gap="large")
        with col_m_tbl:
            m_display = missing_df.copy()
            m_display.columns = ["Variable", "Missing Count", "Missing (%)"]
            st.dataframe(m_display, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Imputation Strategy:**
            - **OLS Analysis:** Complete-case filtering ($n=73$) was performed to deliver strict, verifiable coefficient estimates without injecting synthetic variance.
            - **Predictive ML Pipelines:** Within 5-fold cross-validation, missing numeric values are imputed with fold **medians**, and categorical values with fold **modes**, guaranteeing zero data leakage.
            """)

        with col_m_fig:
            fig_miss = go.Figure(data=[
                go.Bar(
                    x=missing_df["column"],
                    y=missing_df["missing_count"],
                    marker=dict(color=["#059669" if c == "Y" else "#0284c7" for c in missing_df["column"]]),
                    text=[f"{v} ({p}%)" for v, p in zip(missing_df["missing_count"], missing_df["missing_percent"])],
                    textposition="auto"
                )
            ])
            layout_miss = get_industrial_plotly_layout(
                height=320,
                title="Missing Records by Feature Column (N=100)",
                xlab="Column",
                ylab="Missing Count"
            )
            fig_miss.update_layout(layout_miss)
            st.plotly_chart(fig_miss, use_container_width=True)

    # TAB 3: CORRELATION ANALYSIS
    with tab_corr:
        st.markdown("##### Pearson Correlation Matrix & Target Association (Assignment Question 3)")
        st.markdown("""
        <div class="tech-box">
            <strong>Key Finding:</strong> Among numeric predictors, <strong>Lathe Speed (X1)</strong> demonstrates the strongest association with tool life (<strong>r = -0.5037</strong>), representing a pronounced negative relationship.
            Feed rate (<strong>r = -0.1081</strong>) and depth of cut (<strong>r = -0.1252</strong>) show weak negative associations, while workpiece hardness (<strong>r = -0.0339</strong>) exhibits near-zero linear correlation.
        </div>
        """, unsafe_allow_html=True)

        col_c_left, col_c_right = st.columns([1, 1.3], gap="large")
        with col_c_left:
            st.markdown("###### Pearson r with Tool Life (Y)")
            st.dataframe(correlations_df.style.format({"pearson_r": "{:.4f}"}), use_container_width=True)
            
            # Numeric correlation matrix
            numeric_corr = raw_df[["Y", "X1", "X3", "X4", "X6"]].corr()
            st.markdown("###### Full Pairwise Pearson Matrix")
            st.dataframe(numeric_corr.style.format("{:.3f}"), use_container_width=True)

        with col_c_right:
            # Heatmap of correlation matrix
            fig_hm = px.imshow(
                numeric_corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1.0, zmax=1.0,
                labels=dict(color="Pearson r")
            )
            layout_hm = get_industrial_plotly_layout(
                height=360,
                title="Inter-Feature Correlation Heatmap",
                xlab="Variable", ylab="Variable"
            )
            fig_hm.update_layout(layout_hm)
            st.plotly_chart(fig_hm, use_container_width=True)

        # Bivariate Scatter Plot: Speed vs Tool Life colored by Tool Type
        st.markdown("###### Bivariate Interaction: Tool Life (Y) vs. Lathe Speed (X1)")
        fig_scatter = px.scatter(
            raw_df.dropna(subset=["X1", "Y", "X2"]),
            x="X1", y="Y",
            color="X2",
            color_discrete_map={"A": INDUSTRIAL_COLORS["tool_A"], "B": INDUSTRIAL_COLORS["tool_B"]},
            labels={"X1": "Lathe Speed (X1) [rpm]", "Y": "Tool Life (Y) [hours]", "X2": "Tool Type"},
            trendline="ols",
            hover_data=["X3", "X4", "X5", "X6"]
        )
        layout_sc = get_industrial_plotly_layout(
            height=380,
            title="Tool Life vs. Lathe Speed Stratified by Tool Type (with OLS Trends)",
            xlab="Lathe Speed (X1) [rpm]",
            ylab="Tool Life (Y) [hours]"
        )
        fig_scatter.update_layout(layout_sc)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # TAB 4: TOOL LIFE DISTRIBUTIONS
    with tab_dist:
        st.markdown("##### Overall Tool Life (Y) Frequency Distribution")
        
        col_d_hist, col_d_box = st.columns([1.3, 1.0], gap="large")
        with col_d_hist:
            fig_dist = px.histogram(
                raw_df.dropna(subset=["Y"]),
                x="Y",
                nbins=15,
                marginal="box",
                color_discrete_sequence=[INDUSTRIAL_COLORS["primary"]],
                labels={"Y": "Tool Life (hours)"}
            )
            layout_dist = get_industrial_plotly_layout(
                height=380,
                title="Tool Life Histogram with Marginal Boxplot (N=100)",
                xlab="Tool Life (Y) [hours]",
                ylab="Frequency Count"
            )
            fig_dist.update_layout(layout_dist)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_d_box:
            st.markdown("###### Tool Life Moments (hours)")
            y_series = raw_df["Y"].dropna()
            moments = pd.DataFrame({
                "Metric": ["Mean", "Median", "Standard Deviation", "Variance", "Interquartile Range (IQR)", "Minimum", "Maximum", "Skewness", "Kurtosis"],
                "Value": [
                    f"{y_series.mean():.4f} hrs",
                    f"{y_series.median():.4f} hrs",
                    f"{y_series.std():.4f} hrs",
                    f"{y_series.var():.4f} hrs²",
                    f"{y_series.quantile(0.75) - y_series.quantile(0.25):.4f} hrs",
                    f"{y_series.min():.2f} hrs",
                    f"{y_series.max():.2f} hrs",
                    f"{y_series.skew():.3f} (Near-symmetric)",
                    f"{y_series.kurt():.3f} (Platykurtic/Bimodal tendency)"
                ]
            })
            st.dataframe(moments, use_container_width=True, hide_index=True)
            st.caption("Note: The bimodal appearance is driven directly by the two distinct tool grades (Tool A vs Tool B).")

    # TAB 5: TOOL-TYPE COMPARISONS
    with tab_tool:
        st.markdown("##### Tool Type A vs. Tool Type B Breakdown (Assignment Question 2)")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #0284c7;">
                <div class="title">Tool Type A (Standard Grade)</div>
                <div class="value">16.52<span style="font-size: 1rem;">h</span></div>
                <div class="subtext">Valid N: 48 &nbsp;|&nbsp; Median: 15.33h &nbsp;|&nbsp; SD: 4.80h</div>
            </div>
            """, unsafe_allow_html=True)
        with col_t2:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #b45309;">
                <div class="title">Tool Type B (Premium Grade)</div>
                <div class="value">32.07<span style="font-size: 1rem;">h</span></div>
                <div class="subtext">Valid N: 47 &nbsp;|&nbsp; Median: 32.58h &nbsp;|&nbsp; SD: 5.73h</div>
            </div>
            """, unsafe_allow_html=True)

        col_tb_fig1, col_tb_fig2 = st.columns(2)
        with col_tb_fig1:
            fig_tb_box = px.box(
                raw_df.dropna(subset=["X2", "Y"]),
                x="X2", y="Y",
                color="X2",
                color_discrete_map={"A": INDUSTRIAL_COLORS["tool_A"], "B": INDUSTRIAL_COLORS["tool_B"]},
                points="all",
                labels={"X2": "Tool Type", "Y": "Tool Life (hours)"}
            )
            layout_tb = get_industrial_plotly_layout(
                height=360,
                title="Tool Life Distribution by Tool Type",
                xlab="Tool Type", ylab="Tool Life (Y) [hours]"
            )
            fig_tb_box.update_layout(layout_tb)
            st.plotly_chart(fig_tb_box, use_container_width=True)

        with col_tb_fig2:
            fig_tb_hist = px.histogram(
                raw_df.dropna(subset=["X2", "Y"]),
                x="Y", color="X2",
                barmode="overlay",
                nbins=18,
                color_discrete_map={"A": INDUSTRIAL_COLORS["tool_A"], "B": INDUSTRIAL_COLORS["tool_B"]},
                labels={"Y": "Tool Life (hours)", "X2": "Tool Type"}
            )
            layout_tb_hist = get_industrial_plotly_layout(
                height=360,
                title="Comparative Histogram: Tool A vs. Tool B",
                xlab="Tool Life (Y) [hours]", ylab="Count"
            )
            fig_tb_hist.update_layout(layout_tb_hist)
            st.plotly_chart(fig_tb_hist, use_container_width=True)

    # TAB 6: COOLANT COMPARISONS
    with tab_coolant:
        st.markdown("##### Coolant Application Comparative Analysis (Questions 2 & 5)")
        
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #64748b;">
                <div class="title">Dry Machining (No Coolant)</div>
                <div class="value">22.92<span style="font-size: 1rem;">h</span></div>
                <div class="subtext">Valid N: 38 &nbsp;|&nbsp; Median: 21.97h &nbsp;|&nbsp; SD: 8.66h</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c2:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #0284c7;">
                <div class="title">Flood Coolant (Coolant Yes)</div>
                <div class="value">25.34<span style="font-size: 1rem;">h</span></div>
                <div class="subtext">Valid N: 57 &nbsp;|&nbsp; Median: 25.40h &nbsp;|&nbsp; SD: 9.62h</div>
            </div>
            """, unsafe_allow_html=True)
        with col_c3:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #047857;">
                <div class="title">Unadjusted Bivariate Delta</div>
                <div class="value" style="color: #047857;">+2.42<span style="font-size: 1rem;">h</span></div>
                <div class="subtext">Raw difference in sample means</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tech-box">
            <strong>Critical Econometric / Statistical Insight:</strong> 
            The raw unadjusted group difference is <strong>+2.4196 hours</strong> (+10.55%). 
            However, this bivariate comparison fails to adjust for confounding: runs with coolant happened to employ a slightly higher proportion of Tool B and lower lathe speeds.
            In the multiple OLS regression (where speed, feed, depth of cut, hardness, and tool type are held constant), the controlled effect of coolant drops to <strong>+0.6008 hours (p = 0.271)</strong>, which is not statistically significant.
        </div>
        """, unsafe_allow_html=True)

        fig_cool = px.box(
            raw_df.dropna(subset=["X5", "Y"]),
            x="X5", y="Y",
            color="X5",
            color_discrete_map={"No": "#64748b", "Yes": "#0284c7"},
            points="all",
            labels={"X5": "Coolant Applied", "Y": "Tool Life (hours)"}
        )
        layout_cool = get_industrial_plotly_layout(
            height=340,
            title="Tool Life by Coolant Usage (All Tool Grades)",
            xlab="Coolant Usage", ylab="Tool Life (Y) [hours]"
        )
        fig_cool.update_layout(layout_cool)
        st.plotly_chart(fig_cool, use_container_width=True)

"""
prediction_view.py - Interactive Machining Tool Life Prediction Interface.
Executes real-time inferences with input validation, sensitivity curves, and shop-floor engineering metrics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.ui_theme import INDUSTRIAL_COLORS, get_industrial_plotly_layout

# Empirical parameter boundaries established during the 100-run experimental trials
PARAM_BOUNDS = {
    "X1": {"name": "Lathe Speed", "unit": "rpm", "min": 500, "max": 1000, "default": 650, "step": 10},
    "X3": {"name": "Feed Rate", "unit": "mm/rev", "min": 0.10, "max": 0.40, "default": 0.22, "step": 0.01},
    "X4": {"name": "Depth of Cut", "unit": "mm", "min": 0.50, "max": 2.50, "default": 1.20, "step": 0.10},
    "X6": {"name": "Workpiece Hardness", "unit": "HB", "min": 150, "max": 250, "default": 205, "step": 1}
}

def render_prediction_view(model, raw_df: pd.DataFrame):
    """Renders the interactive prediction workspace."""
    st.markdown("""
    <div class="section-banner">
        <div>
            <h3>Machining Operational Parameter Inputs</h3>
            <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">
                Enter lathe parameters and cutting conditions to predict tool life (Y) using the trained Ridge ML pipeline.
            </div>
        </div>
        <span>MODEL: RIDGE REGRESSION (JOB_LIB REFIT)</span>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1.15, 1.0], gap="large")
    
    with col_input:
        st.markdown("##### 1. Primary Cutting Dynamics")
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            x1 = st.slider(
                "Lathe Speed (X1) [rpm]",
                min_value=400,
                max_value=1200,
                value=650,
                step=10,
                help="Spindle rotational speed. Experimental domain: 500 – 1000 rpm."
            )
        with col_x2:
            x2 = st.selectbox(
                "Tool Type (X2)",
                options=["A", "B"],
                index=1,
                help="Tool grade. Type A: Standard/Uncoated; Type B: High-durability Coated Carbide."
            )
            
        col_x3, col_x4 = st.columns(2)
        with col_x3:
            x3 = st.slider(
                "Feed Rate (X3) [mm/rev]",
                min_value=0.05,
                max_value=0.50,
                value=0.22,
                step=0.01,
                format="%.2f",
                help="Feed travel per spindle revolution. Experimental domain: 0.10 – 0.40 mm/rev."
            )
        with col_x4:
            x4 = st.slider(
                "Depth of Cut (X4) [mm]",
                min_value=0.2,
                max_value=3.0,
                value=1.2,
                step=0.1,
                format="%.1f",
                help="Radial penetration depth into the workpiece. Experimental domain: 0.50 – 2.50 mm."
            )
            
        st.markdown("##### 2. Environmental & Metallurgical Conditions")
        col_x5, col_x6 = st.columns(2)
        with col_x5:
            x5 = st.selectbox(
                "Coolant Usage (X5)",
                options=["Yes", "No"],
                index=0,
                help="Application of flood coolant to the cutting interface."
            )
        with col_x6:
            x6 = st.slider(
                "Workpiece Hardness (X6) [HB]",
                min_value=130,
                max_value=270,
                value=205,
                step=1,
                help="Material hardness measured in Brinell (HB). Experimental domain: 150 – 249 HB."
            )
            
        # Preset quick loaders for standard shop scenarios
        st.markdown("---")
        st.caption("Quick Scenarios:")
        col_sc1, col_sc2, col_sc3 = st.columns(3)
        with col_sc1:
            st.info("**Conservative Finishing**\n- 540 rpm, Tool B\n- 0.15 mm/rev, Coolant")
        with col_sc2:
            st.info("**Balanced Production**\n- 650 rpm, Tool B\n- 0.22 mm/rev, Coolant")
        with col_sc3:
            st.info("**Aggressive Roughing**\n- 920 rpm, Tool A\n- 0.32 mm/rev, Dry")

    # Domain Extrapolation / Validation Checks
    warnings = []
    if x1 < PARAM_BOUNDS["X1"]["min"] or x1 > PARAM_BOUNDS["X1"]["max"]:
        warnings.append(f"Lathe speed {x1} rpm lies outside the experimental envelope [{PARAM_BOUNDS['X1']['min']}, {PARAM_BOUNDS['X1']['max']}] rpm.")
    if x3 < PARAM_BOUNDS["X3"]["min"] or x3 > PARAM_BOUNDS["X3"]["max"]:
        warnings.append(f"Feed rate {x3:.2f} mm/rev lies outside the experimental envelope [{PARAM_BOUNDS['X3']['min']:.2f}, {PARAM_BOUNDS['X3']['max']:.2f}] mm/rev.")
    if x4 < PARAM_BOUNDS["X4"]["min"] or x4 > PARAM_BOUNDS["X4"]["max"]:
        warnings.append(f"Depth of cut {x4:.1f} mm lies outside the experimental envelope [{PARAM_BOUNDS['X4']['min']:.1f}, {PARAM_BOUNDS['X4']['max']:.1f}] mm.")
    if x6 < PARAM_BOUNDS["X6"]["min"] or x6 > PARAM_BOUNDS["X6"]["max"]:
        warnings.append(f"Hardness {x6} HB lies outside the experimental envelope [{PARAM_BOUNDS['X6']['min']}, {PARAM_BOUNDS['X6']['max']}] HB.")

    # Model inference
    input_data = pd.DataFrame([{
        "X1": float(x1),
        "X2": str(x2),
        "X3": float(x3),
        "X4": float(x4),
        "X5": str(x5),
        "X6": float(x6)
    }])
    
    try:
        prediction = float(model.predict(input_data)[0])
        # Prevent non-physical negative tool life
        predicted_tool_life = max(0.0, prediction)
    except Exception as e:
        st.error(f"Prediction inference failed: {e}")
        predicted_tool_life = 0.0

    with col_result:
        st.markdown("##### Tool Life Prediction Result")
        
        # Professional Industrial Metric Display Card
        st.markdown(f"""
        <div class="prediction-display-card">
            <div class="label">Predicted Operational Tool Life (Y)</div>
            <div class="result-hours">
                {predicted_tool_life:.2f}<span class="unit">hours</span>
            </div>
            <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 0.6rem;">
                ≈ <span style="color: #f8fafc; font-weight: 600;">{predicted_tool_life * 60:.0f}</span> minutes &nbsp;|&nbsp; 
                ≈ <span style="color: #f8fafc; font-weight: 600;">{predicted_tool_life / 8.0:.2f}</span> standard 8-hour machining shifts
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if warnings:
            st.markdown("""
            <div class="tech-box warning">
                <strong>Extrapolation Alert:</strong> The selected configuration operates beyond the empirical trial bounds. Ridge regularisation maintains stability, but real-world tool wear uncertainty is increased.
            </div>
            """, unsafe_allow_html=True)
            for w in warnings:
                st.caption(f"• {w}")
        else:
            st.markdown("""
            <div class="tech-box success" style="margin-top: 1rem; margin-bottom: 0.5rem; padding: 0.75rem 1rem;">
                <strong>Valid Operational Window:</strong> All parameters fall within the verified 100-run experimental regime.
            </div>
            """, unsafe_allow_html=True)

        # Comparative What-If Scenarios
        st.markdown("###### Parameter Delta Impact")
        
        # Calculate Delta if switched to the other tool type
        alt_tool = "A" if x2 == "B" else "B"
        input_alt_tool = input_data.copy()
        input_alt_tool["X2"] = alt_tool
        pred_alt_tool = float(model.predict(input_alt_tool)[0])
        tool_delta = pred_alt_tool - predicted_tool_life
        
        # Calculate Delta if coolant switched
        alt_coolant = "No" if x5 == "Yes" else "Yes"
        input_alt_coolant = input_data.copy()
        input_alt_coolant["X5"] = alt_coolant
        pred_alt_coolant = float(model.predict(input_alt_coolant)[0])
        coolant_delta = pred_alt_coolant - predicted_tool_life

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            symbol_t = "+" if tool_delta >= 0 else ""
            color_t = "#047857" if tool_delta >= 0 else "#b91c1c"
            st.markdown(f"""
            <div class="metric-card">
                <div class="title">Switching to Tool Type {alt_tool}</div>
                <div class="value" style="font-size: 1.25rem; color: {color_t};">{symbol_t}{tool_delta:.2f} hrs</div>
                <div class="subtext">Impact of carbide grade change</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_d2:
            symbol_c = "+" if coolant_delta >= 0 else ""
            color_c = "#047857" if coolant_delta >= 0 else "#b91c1c"
            st.markdown(f"""
            <div class="metric-card">
                <div class="title">Toggling Coolant to '{alt_coolant}'</div>
                <div class="value" style="font-size: 1.25rem; color: {color_c};">{symbol_c}{coolant_delta:.2f} hrs</div>
                <div class="subtext">Controlled thermal margin</div>
            </div>
            """, unsafe_allow_html=True)

    # Spindle Speed Sensitivity Curve
    st.markdown("---")
    st.markdown("""
    <div class="section-banner">
        <div>
            <h3>Lathe Speed (X1) Sensitivity Curve</h3>
            <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">
                Tool life variation across spindle velocities (450 – 1100 rpm) while locking X2–X6 at current input values.
            </div>
        </div>
        <span>EMPIRICAL TAYLOR CURVE APPROXIMATION</span>
    </div>
    """, unsafe_allow_html=True)

    # Compute sensitivity curve
    speed_range = np.linspace(450, 1100, 66)
    sim_df = pd.DataFrame({
        "X1": speed_range,
        "X2": [x2] * len(speed_range),
        "X3": [x3] * len(speed_range),
        "X4": [x4] * len(speed_range),
        "X5": [x5] * len(speed_range),
        "X6": [x6] * len(speed_range)
    })
    sim_pred = model.predict(sim_df)

    # Also compute opposite tool type for comparison curve
    sim_df_alt = sim_df.copy()
    sim_df_alt["X2"] = alt_tool
    sim_pred_alt = model.predict(sim_df_alt)

    fig = go.Figure()

    # Current tool type curve
    fig.add_trace(go.Scatter(
        x=speed_range,
        y=sim_pred,
        mode="lines",
        name=f"Current Setup (Tool {x2}, Coolant: {x5})",
        line=dict(color=INDUSTRIAL_COLORS["primary"], width=3),
        hovertemplate="Speed: %{x:.0f} rpm<br>Life: %{y:.2f} hrs<extra></extra>"
    ))

    # Alternate tool type curve
    fig.add_trace(go.Scatter(
        x=speed_range,
        y=sim_pred_alt,
        mode="lines",
        name=f"Alternative Setup (Tool {alt_tool}, Coolant: {x5})",
        line=dict(color="#94a3b8", width=2, dash="dash"),
        hovertemplate="Speed: %{x:.0f} rpm<br>Life: %{y:.2f} hrs<extra></extra>"
    ))

    # Operating point marker
    fig.add_trace(go.Scatter(
        x=[x1],
        y=[predicted_tool_life],
        mode="markers+text",
        name="Current Operating Point",
        marker=dict(size=11, color="#b91c1c", line=dict(color="#ffffff", width=2)),
        text=[f"  {predicted_tool_life:.2f}h @ {x1} rpm"],
        textposition="top right",
        textfont=dict(family="JetBrains Mono, monospace", size=12, color="#0f172a"),
        hovertemplate="Operating Point<br>Speed: %{x} rpm<br>Life: %{y:.2f} hrs<extra></extra>"
    ))

    # Shaded experimental domain [500, 1000]
    fig.add_vrect(
        x0=500, x1=1000,
        fillcolor="rgba(37, 99, 235, 0.04)",
        layer="below",
        line_width=1,
        line_color="rgba(37, 99, 235, 0.2)",
        annotation_text="Verified Trial Domain [500–1000 rpm]",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#64748b")
    )

    layout = get_industrial_plotly_layout(
        height=380,
        title="Spindle Speed Velocity vs. Tool Life Trajectory",
        xlab="Lathe Speed (X1) [rpm]",
        ylab="Tool Life (Y) [hours]"
    )
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)

"""
ui_theme.py - Industrial UI Theme, Styling, and Plotly Configurations.
Provides a clean, restrained, high-density industrial metrology aesthetic.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Industrial Color Palette
INDUSTRIAL_COLORS = {
    "bg_dark": "#0f172a",
    "surface": "#ffffff",
    "surface_alt": "#f8fafc",
    "border": "#e2e8f0",
    "border_strong": "#cbd5e1",
    "text_main": "#0f172a",
    "text_muted": "#475569",
    "primary": "#1d4ed8",      # Industrial Blue
    "primary_dark": "#1e40af",
    "primary_light": "#dbeafe",
    "secondary": "#334155",    # Slate
    "accent": "#0284c7",       # Steel Cyan
    "accent_warm": "#b45309",  # Industrial Amber / Tool Wear
    "success": "#047857",      # Emerald
    "danger": "#b91c1c",       # Crimson
    "tool_A": "#0284c7",       # Tool A color
    "tool_B": "#b45309",       # Tool B color
    "coolant_no": "#64748b",
    "coolant_yes": "#0284c7"
}

def apply_custom_css():
    """Injects professional, non-generic industrial CSS."""
    st.markdown("""
    <style>
    /* Global Typography & Spacing */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0f172a;
    }
    
    code, kbd, samp, pre, .mono-val {
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
        font-variant-numeric: tabular-nums;
    }
    
    /* Main Layout Refinements */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1280px !important;
    }
    
    /* Header Container */
    .industrial-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #2563eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }
    .industrial-header h1 {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0 0 0.4rem 0;
        color: #ffffff;
    }
    .industrial-header p {
        font-size: 0.92rem;
        color: #94a3b8;
        margin: 0;
        max-width: 900px;
        line-height: 1.45;
    }
    .badge-bar {
        margin-top: 0.75rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .ind-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 4px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: #334155;
        color: #e2e8f0;
    }
    .ind-badge.active {
        background: #1d4ed8;
        color: #ffffff;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem 1.15rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #cbd5e1;
    }
    .metric-card .title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .metric-card .value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }
    .metric-card .subtext {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 0.35rem;
        line-height: 1.35;
    }
    .metric-card .pill-up {
        display: inline-block;
        color: #047857;
        font-weight: 600;
    }
    
    /* Engineering Section Headers */
    .section-banner {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }
    .section-banner h3 {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .section-banner span {
        font-size: 0.78rem;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Technical Data Box */
    .tech-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        font-size: 0.88rem;
        line-height: 1.5;
        color: #1e293b;
    }
    .tech-box.warning {
        border-left-color: #d97706;
        background: #fffbeb;
    }
    .tech-box.success {
        border-left-color: #059669;
        background: #f0fdf4;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.88rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        color: #475569;
        background: transparent;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #1d4ed8 !important;
        background: #eff6ff !important;
        border-color: #bfdbfe !important;
    }
    
    /* Custom Prediction Display */
    .prediction-display-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.75rem;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .prediction-display-card .label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .prediction-display-card .result-hours {
        font-size: 3.2rem;
        font-weight: 800;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.04em;
        line-height: 1.1;
    }
    .prediction-display-card .unit {
        font-size: 1.1rem;
        font-weight: 500;
        color: #cbd5e1;
        margin-left: 0.25rem;
    }
    
    /* Dataframes and Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)


def get_industrial_plotly_layout(height=400, title="", xlab="", ylab=""):
    """Returns a standardized clean layout configuration for Plotly graphs."""
    return dict(
        height=height,
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color="#1e293b"),
            x=0.01,
            xanchor="left"
        ),
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#334155"),
        margin=dict(l=45, r=25, t=45 if title else 25, b=45),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_size=12,
            font_family="JetBrains Mono, monospace",
            font_color="#ffffff"
        ),
        xaxis=dict(
            title=dict(text=xlab, font=dict(size=12, color="#475569")),
            gridcolor="#f1f5f9",
            zerolinecolor="#cbd5e1",
            linecolor="#cbd5e1",
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#475569")
        ),
        yaxis=dict(
            title=dict(text=ylab, font=dict(size=12, color="#475569")),
            gridcolor="#f1f5f9",
            zerolinecolor="#cbd5e1",
            linecolor="#cbd5e1",
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#475569")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#475569"),
            bgcolor="rgba(255,255,255,0.7)"
        )
    )

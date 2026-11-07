"""
assignment_qa_view.py - Original Assignment Questions & Calculated Solutions.
Presents the five original examination questions with their formal numerical solutions,
reproducible formulas, and engineering interpretations.
"""
import streamlit as st
import pandas as pd
from modules.ui_theme import INDUSTRIAL_COLORS

def render_assignment_qa_view(questions_data: list[dict]):
    """Renders the five assignment questions with verified calculated answers."""
    
    st.markdown("""
    <div class="section-banner">
        <div>
            <h3>Original Examination Questions & Verified Solutions</h3>
            <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">
                Formal answers directly aligned with the workbook sheet <code>Questions</code> in <code>TOOL_LIFE_DATA_100_ANSWERED.xlsx</code>.
            </div>
        </div>
        <span>SOURCE: TOOL_LIFE_DATA_100_ANSWERED.XLSX</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tech-box">
        <strong>Academic & Industrial Context:</strong> 
        The five questions evaluate the progression from initial data quality assessment (missingness, distributions) 
        to bivariate association (Pearson correlation) and multivariate causal/controlled modeling (Full vs. Reduced OLS, Simpson's Paradox / confounding with coolant).
    </div>
    """, unsafe_allow_html=True)

    for item in questions_data:
        qid = item["id"]
        q_title = item["title"]
        q_text = item["question"]
        ans_text = item["answer"]
        key_metric = item["key_metric"]
        details = item.get("details", {})
        interp = item.get("interpretation", "")

        with st.container():
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #1d4ed8; text-transform: uppercase; letter-spacing: 0.08em; background: #eff6ff; padding: 0.2rem 0.5rem; border-radius: 4px;">
                        QUESTION {qid}
                    </span>
                    <span style="font-size: 0.78rem; font-weight: 600; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                        {key_metric}
                    </span>
                </div>
                <div style="font-size: 1.02rem; font-weight: 700; color: #0f172a; margin-bottom: 0.75rem;">
                    {q_text}
                </div>
                <div style="background: #f8fafc; border-left: 4px solid #1d4ed8; padding: 0.85rem 1rem; border-radius: 4px; font-size: 0.92rem; color: #1e293b; font-family: 'JetBrains Mono', monospace; line-height: 1.5; margin-bottom: 0.75rem;">
                    <strong>Formal Answer:</strong><br>
                    {ans_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"🔍 Detailed Solution Breakdown & Engineering Commentary for Question {qid}"):
                col_d_left, col_d_right = st.columns([1, 1], gap="medium")
                with col_d_left:
                    st.markdown("**Structured Values:**")
                    detail_df = pd.DataFrame(list(details.items()), columns=["Parameter", "Calculated Value"])
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                with col_d_right:
                    st.markdown("**Engineering & Statistical Rationale:**")
                    st.markdown(interp)

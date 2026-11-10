
import os
import sys
import streamlit as st
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(BASE_DIR, "app")

sys.path.insert(0, APP_DIR)

from model_service import load_pipeline, analyze_source_code

st.set_page_config(
    page_title="Smart Contract Security Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom styling
with open(os.path.join(APP_DIR, "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load models once
@st.cache_resource
def initialize_pipeline():
    return load_pipeline()


try:
    package, extractor, decision_engine = initialize_pipeline()
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    pipeline_error = str(e)


# Sidebar
with st.sidebar:
    st.markdown("# 🛡️ SC ANALYZER")
    st.caption("SMART CONTRACT SECURITY")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Contract Analyzer", "Model Information", "About Project"]
    )

    st.divider()

    if pipeline_ready:
        st.success("8 ML classifiers loaded")
    else:
        st.error("Model loading failed")

    st.caption("VERSION 1.0.0")


# Dashboard
if page == "Dashboard":

    st.title("Smart Contract Security Analyzer")
    st.caption("AI-powered vulnerability classification for Solidity contracts.")

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("ML Classifiers", "08")
    c2.metric("Model Features", "306")
    c3.metric("Supported Files", ".sol")

    st.markdown("## Analysis workflow")

    c1, c2, c3, c4 = st.columns(4)
    c1.info("① Upload Contract")
    c2.info("② Extract Features")
    c3.info("③ Run Classifiers")
    c4.info("④ View Results")

    st.divider()

    st.markdown("### Start a security analysis")

    if st.button("Analyze a Contract", use_container_width=True):
        st.session_state["page"] = "Contract Analyzer"
        st.rerun()


# Contract Analyzer
elif page == "Contract Analyzer":

    st.title("Contract Analyzer")
    st.caption("Upload a Solidity source file to analyze potential vulnerabilities.")

    uploaded_file = st.file_uploader(
        "Upload Solidity Contract",
        type=["sol"],
        help="Upload a Solidity source file with .sol extension."
    )

    if uploaded_file:

        source_code = uploaded_file.getvalue().decode(
            "utf-8", errors="replace"
        )

        st.success(f"File uploaded: {uploaded_file.name}")

        with st.expander("View Solidity Source Code"):
            st.code(source_code, language="solidity")

        if st.button("⚡ Analyze Contract", use_container_width=True):

            if not pipeline_ready:
                st.error(f"Model loading error: {pipeline_error}")

            elif not source_code.strip():
                st.error("The uploaded file is empty.")

            else:
                try:
                    with st.spinner("Extracting features and analyzing contract..."):

                        results, warnings = analyze_source_code(
                            source_code,
                            package,
                            extractor,
                            decision_engine
                        )

                    result = results.iloc[0]

                    st.session_state["analysis_result"] = result.to_dict()
                    st.session_state["analysis_filename"] = uploaded_file.name
                    st.session_state["analysis_warnings"] = warnings

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    # Display results
    if "analysis_result" in st.session_state:

        result = st.session_state["analysis_result"]
        warnings = st.session_state.get("analysis_warnings", [])

        st.divider()
        st.markdown("## Analysis Results")

        st.caption(
            f"Contract: {st.session_state['analysis_filename']}"
        )

        status = result["final_status"]

        status_colors = {
            "Safe-only": "#00E676",
            "Vulnerability Detected": "#FF4D6D",
            "Model Disagreement": "#FBBF24",
            "No Targeted Vulnerability Detected": "#00C6FF"
        }

        status_color = status_colors.get(status, "#A78BFA")

        st.markdown(
            f"""
            <div style="
                background:#151C2B;
                border:1px solid {status_color};
                border-radius:12px;
                padding:22px;
                margin:10px 0 20px 0;
            ">
                <div style="color:#9CA3AF;font-size:13px;">
                    FINAL CLASSIFICATION
                </div>
                <div style="
                    color:{status_color};
                    font-size:27px;
                    font-weight:700;
                    margin-top:8px;
                ">
                    {status}
                </div>
                <div style="color:#CBD5E1;margin-top:8px;">
                    {result['number_of_vulnerabilities']}
                    vulnerability indicators detected
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        vulnerability_names = [
            "Access Control",
            "Arithmetic",
            "Reentrancy",
            "Unchecked Calls",
            "Locked Ether",
            "Bad Randomness",
            "Double Spending"
        ]

        probabilities = [
            float(result[f"{name}_probability"])
            for name in vulnerability_names
        ]

        predictions = [
            int(result[f"{name}_prediction"])
            for name in vulnerability_names
        ]

        # Probability chart
        st.markdown("### Vulnerability Model Scores")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=[p * 100 for p in probabilities],
            y=vulnerability_names,
            orientation="h",
            marker_color=[
                "#FF4D6D" if pred else "#00C6FF"
                for pred in predictions
            ],
            text=[f"{p * 100:.1f}%" for p in probabilities],
            textposition="outside"
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#080B12",
            plot_bgcolor="#080B12",
            font_color="#E2E8F0",
            xaxis_title="Model score (%)",
            xaxis=dict(range=[0, 110]),
            yaxis=dict(autorange="reversed"),
            height=400,
            margin=dict(l=10, r=40, t=20, b=30)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Vulnerability cards
        st.markdown("### Individual Classifier Results")

        for i in range(0, len(vulnerability_names), 2):

            cols = st.columns(2)

            for j, col in enumerate(cols):

                index = i + j

                if index >= len(vulnerability_names):
                    break

                name = vulnerability_names[index]
                probability = probabilities[index]
                prediction = predictions[index]

                with col:
                    if prediction:
                        st.error(
                            f"**{name}** — DETECTED\n\n"
                            f"Model score: {probability:.2%}"
                        )
                    else:
                        st.info(
                            f"**{name}** — NOT DETECTED\n\n"
                            f"Model score: {probability:.2%}"
                        )

        # Safe classifier
        safe_probability = float(result["Safe_probability"])
        safe_prediction = int(result["Safe_prediction"])

        st.markdown("### Safe Contract Classifier")

        if safe_prediction:
            st.success(
                f"Safe classifier activated — score: {safe_probability:.2%}"
            )
        else:
            st.warning(
                f"Safe classifier not activated — score: {safe_probability:.2%}"
            )

        if warnings:
            st.warning("Feature extraction warnings:")
            for warning in warnings:
                st.write(warning)
        else:
            st.caption("Feature extraction completed without warnings.")

        st.divider()

        st.caption(
            "Model scores are classifier outputs, not calibrated probabilities "
            "of actual security vulnerabilities. This tool does not replace "
            "a professional security audit."
        )


# Model Information
elif page == "Model Information":

    st.title("Model Architecture")
    st.caption("Hybrid machine learning classification system.")

    classifiers = [
        "Access Control — XGBoost",
        "Arithmetic — XGBoost",
        "Reentrancy — Random Forest",
        "Unchecked Calls — XGBoost",
        "Locked Ether — XGBoost",
        "Bad Randomness — XGBoost",
        "Double Spending — XGBoost",
        "Safe — XGBoost"
    ]

    for classifier in classifiers:
        st.markdown(f"- 🟢 {classifier}")

    st.markdown("### Feature Extraction")
    st.write("322 extracted features, of which 306 are used by the trained models.")

    st.markdown("### Decision Engine")
    st.write("""
    The decision engine combines the eight binary predictions into four
    possible final classifications:
    - Safe-only
    - Vulnerability Detected
    - Model Disagreement
    - No Targeted Vulnerability Detected
    """)


# About Project
elif page == "About Project":

    st.title("About the Project")

    st.markdown("""
    This project applies machine learning to identify potential
    vulnerability patterns in Solidity smart contracts.

    **Core components**
    - Solidity source-code feature extraction
    - Eight binary classification models
    - Hybrid prediction architecture
    - Rule-based decision engine

    **Technology stack**
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - XGBoost
    - Random Forest
    - Plotly

    **Disclaimer:** Automated predictions are indicators, not proof
    that a contract is secure or vulnerable.
    """)

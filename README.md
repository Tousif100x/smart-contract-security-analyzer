# Smart Contract Security Analyzer

A Streamlit-based machine learning application for identifying
potential vulnerability patterns in Solidity smart contracts.

## Features

- Solidity source-code upload
- Automated static feature extraction
- Eight binary ML classifiers
- Hybrid classification decision engine
- Interactive Plotly visualizations

## Architecture

Solidity Source Code
→ Feature Extraction (322 features)
→ Selection of 306 Model Features
→ Eight ML Classifiers
→ Decision Engine
→ Results Dashboard

## Technology Stack

Python, Streamlit, Pandas, Scikit-learn, XGBoost,
Random Forest and Plotly.

## Run Locally

Install dependencies:

    pip install -r requirements.txt

Start the application:

    streamlit run app/app.py

## Limitations

Model scores are classifier outputs, not calibrated probabilities
of actual vulnerabilities. Predictions are not a substitute for
a professional smart contract security audit.
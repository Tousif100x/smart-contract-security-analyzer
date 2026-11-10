
import os
import sys
import joblib
import pandas as pd
import importlib.util

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "final_models")

MODEL_PATH = os.path.join(
    MODEL_DIR, "hybrid_vulnerability_classifier.joblib"
)

EXTRACTOR_PATH = os.path.join(
    BASE_DIR, "extract_contract_features.py"
)

DECISION_PATH = os.path.join(
    MODEL_DIR, "decision_engine.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pipeline():
    package = joblib.load(MODEL_PATH)

    extractor = load_module(
        "feature_extractor",
        EXTRACTOR_PATH
    )

    decision_engine = load_module(
        "decision_engine",
        DECISION_PATH
    )

    return package, extractor, decision_engine


def analyze_source_code(source_code, package, extractor, decision_engine):

    # Extract features
    features, warnings = extractor.extract_features(source_code)

    # Create a one-row DataFrame
    feature_df = pd.DataFrame([features])

    # Select the exact features used during training
    expected_features = package["feature_columns"]

    missing = [
        col for col in expected_features
        if col not in feature_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {len(missing)} required model features."
        )

    X = feature_df[expected_features]

    # Run all classifiers and the decision engine
    results = decision_engine.analyze_contract(
        X,
        package=package
    )

    return results, warnings

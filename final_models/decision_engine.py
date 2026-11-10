
import pandas as pd


VULNERABILITY_NAMES = [
    "Access Control",
    "Arithmetic",
    "Reentrancy",
    "Unchecked Calls",
    "Locked Ether",
    "Bad Randomness",
    "Double Spending"
]


def predict_vulnerabilities(X, package):

    models = package["models"]
    thresholds = package["thresholds"]
    features = package["feature_columns"]

    X = X[features]
    results = pd.DataFrame(index=X.index)

    for name, model in models.items():

        probabilities = model.predict_proba(X)[:, 1]
        threshold = thresholds[name]

        results[f"{name}_probability"] = probabilities
        results[f"{name}_prediction"] = (
            probabilities >= threshold
        ).astype(int)

    return results


def classify_contract(prediction_row):

    safe = prediction_row["Safe_prediction"] == 1

    vulnerabilities = [
        name for name in VULNERABILITY_NAMES
        if prediction_row[f"{name}_prediction"] == 1
    ]

    if safe and vulnerabilities:
        status = "Model Disagreement"

    elif safe:
        status = "Safe-only"

    elif vulnerabilities:
        status = "Vulnerability Detected"

    else:
        status = "No Targeted Vulnerability Detected"

    return {
        "final_status": status,
        "detected_vulnerabilities": vulnerabilities,
        "number_of_vulnerabilities": len(vulnerabilities)
    }


def analyze_contract(X, package):

    predictions = predict_vulnerabilities(X, package)

    decisions = []

    for idx, row in predictions.iterrows():

        decision = classify_contract(row)
        decision["contract_index"] = idx

        decisions.append(decision)

    decisions_df = pd.DataFrame(decisions).set_index(
        "contract_index"
    )

    return predictions.join(decisions_df)

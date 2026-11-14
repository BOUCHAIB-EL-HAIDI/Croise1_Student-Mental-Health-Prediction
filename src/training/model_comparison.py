
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data.data_loader import load_dataset
from src.preprocessing.data_preprocessing import split_data
from src.training.model_training import train_models


REPORT_DIR = PROJECT_ROOT / "reports" / "model_comparison"


def compare_models(trained_models, X_train, y_train):

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision_weighted",
        "Recall": "recall_weighted",
        "F1-Score": "f1_weighted",
        "ROC-AUC": "roc_auc_ovr_weighted"
    }

    results = {}

    for name, model in trained_models.items():

        print(f"\n--- Comparing {name} ---")

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results[name] = {}

        for metric in scoring:

            values = scores[f"test_{metric}"]

            results[name][f"{metric} Mean"] = values.mean()
            results[name][f"{metric} Std"] = values.std()

    results_df = pd.DataFrame(results).T

    results_df = results_df.sort_values(
        by="F1-Score Mean",
        ascending=False
    )

    results_path = REPORT_DIR / "comparison_results.csv"

    results_df.to_csv(
        results_path,
        index_label="Model"
    )

    print("\n--- Model Comparison Results ---")
    print(results_df.round(4).to_string())

    print(f"\nComparison report saved to: {results_path}")

    return results_df


def main():

    print("--- Model Comparison ---")

    df = load_dataset()

    X_train, X_test, y_train, y_test = split_data(df)

    trained_models = train_models(X_train, y_train)

    compare_models(
        trained_models,
        X_train,
        y_train
    )


if __name__ == "__main__":
    main()

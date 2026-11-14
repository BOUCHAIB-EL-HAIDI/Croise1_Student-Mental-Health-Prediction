
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.data.data_loader import load_dataset
from src.preprocessing.data_preprocessing import split_data


MODEL_PATH = PROJECT_ROOT / "models" / "best_model_pipeline.joblib"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, average="weighted"
        ),
        "Recall": recall_score(
            y_test, y_pred, average="weighted"
        ),
        "F1-Score": f1_score(
            y_test, y_pred, average="weighted"
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="weighted"
        )
    }

    print("\n--- Final Test Results ---")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n--- Classification Report ---")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    print(classification_report(y_test, y_pred))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / "classification_report.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        REPORT_DIR / "evaluation_metrics.csv",
        index=False
    )

    return y_pred, metrics


def plot_confusion_matrix(model, y_test, y_pred):

    labels = model.classes_

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print("\n--- Confusion Matrix ---")
    print(cm)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.show()


def analyze_errors(X_test, y_test, y_pred):

    errors = X_test.copy()

    errors["Actual"] = y_test
    errors["Predicted"] = y_pred

    errors = errors[
        errors["Actual"] != errors["Predicted"]
    ]

    print("\n--- Error Analysis ---")
    print("Total test samples:", len(X_test))
    print("Incorrect predictions:", len(errors))
    print("Error rate:", round(len(errors) / len(X_test) * 100, 2), "%")

    print("\n--- Misclassification Counts ---")
    print(
        errors.groupby(["Actual", "Predicted"]).size()
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    errors.to_csv(
        REPORT_DIR / "misclassified_samples.csv",
        index=False
    )


def main():

    print("--- Final Model Evaluation ---")

    df = load_dataset()

    X_train, X_test, y_train, y_test = split_data(df)

    model = joblib.load(MODEL_PATH)

    print("Loaded model:", MODEL_PATH)
    print("Test samples:", len(X_test))

    y_pred, metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    plot_confusion_matrix(
        model,
        y_test,
        y_pred
    )

    analyze_errors(
        X_test,
        y_test,
        y_pred
    )

    print("\nEvaluation completed.")
    print("Reports saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()

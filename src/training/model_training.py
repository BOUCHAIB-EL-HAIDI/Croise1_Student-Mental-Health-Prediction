
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_dataset
from src.preprocessing.data_preprocessing import (
    split_data,
    create_preprocessor
)


def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "SVM": SVC(
            probability=True,
            random_state=42
        )
    }

    trained_models = {}

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", create_preprocessor(X_train)),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        print(f"{name}:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Number of features: {pipeline.n_features_in_}")
        print(f"  Classes learned: {pipeline.classes_}")

        trained_models[name] = pipeline

        print(f"{name} trained successfully.")

    return trained_models


def main():
    print("--- Model Training ---")

    df = load_dataset()

    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    trained_models = train_models(X_train, y_train)

    print("\n--- Training Completed ---")
    print(f"Total models trained: {len(trained_models)}")


if __name__ == "__main__":
    main()

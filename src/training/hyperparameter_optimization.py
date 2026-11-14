
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data.data_loader import load_dataset
from src.preprocessing.data_preprocessing import split_data
from src.training.model_training import train_models


REPORT_DIR = PROJECT_ROOT / "reports" / "model_optimization"
MODEL_DIR = PROJECT_ROOT / "models"


def optimize_models(trained_models, X_train, y_train):

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    param_grids = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"]
        },

        "SVM": {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"],
            "classifier__class_weight": [None, "balanced"]
        },

        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"]
        }
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    optimized_models = {}
    results = []

    for name, model in trained_models.items():

        print(f"\n--- Optimizing {name} ---")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        optimized_models[name] = grid_search.best_estimator_

        results.append({
            "Model": name,
            "Best F1-Score": grid_search.best_score_,
            "Best Parameters": grid_search.best_params_
        })

        print("Best F1-Score:", round(grid_search.best_score_, 4))
        print("Best Parameters:", grid_search.best_params_)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Best F1-Score",
        ascending=False
    )

    results_path = REPORT_DIR / "optimization_results.csv"

    results_df.to_csv(results_path, index=False)

    print("\n--- Optimization Results ---")
    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {results_path}")






    best_model_name = results_df.iloc[0]["Model"]
    best_model = optimized_models[best_model_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "best_model_pipeline.joblib"

    joblib.dump(best_model, model_path)

    print("\n--- Best Model ---")
    print("Model:", best_model_name)
    print("F1-Score:", round(results_df.iloc[0]["Best F1-Score"], 4))
    print("Saved to:", model_path)

    return optimized_models, best_model_name, best_model


def main():

    print("--- Hyperparameter Optimization ---")

    df = load_dataset()

    X_train, X_test, y_train, y_test = split_data(df)

    trained_models = train_models(X_train, y_train)

    optimize_models(
        trained_models,
        X_train,
        y_train
    )


if __name__ == "__main__":
    main()

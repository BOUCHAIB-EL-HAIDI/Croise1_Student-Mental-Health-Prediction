
import numpy as np

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data.data_loader import load_dataset
from src.preprocessing.data_preprocessing import (
    split_data,
    create_preprocessor
)


TARGET = "Mental_Health_Risk"
CLASSES = ["High", "Low", "Medium"]


def prepare_data():
    df = load_dataset()

    X_train, X_test, y_train, y_test = split_data(df)

    X_train = X_train.drop(
        columns=["Student_ID"],
        errors="ignore"
    )

    return X_train, y_train


def analyze_neighborhood_purity(X, y):
    print("\n" + "=" * 60)
    print("          KNN NEIGHBORHOOD PURITY ANALYSIS")
    print("=" * 60)

    preprocessor = create_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    neighbors_model = NearestNeighbors(
        n_neighbors=5,
        metric="euclidean"
    )

    neighbors_model.fit(X_processed)

    distances, indices = neighbors_model.kneighbors()

    y_array = np.asarray(y)

    neighbor_labels = y_array[indices]

    same_class = neighbor_labels == y_array[:, None]

    purity = same_class.mean(axis=1)

    mixed = (~same_class).any(axis=1)

    print("\nOverall results:")
    print(f"Average neighborhood purity: {purity.mean():.4f}")
    print(f"Mixed neighborhoods: {mixed.mean() * 100:.2f}%")

    print("\nResults by class:")

    for class_name in CLASSES:
        mask = y_array == class_name

        class_purity = purity[mask]
        class_mixed = mixed[mask]

        print(f"\n{class_name}:")
        print(
            f"Average purity: "
            f"{class_purity.mean():.4f}"
        )
        print(
            f"Mixed neighborhoods: "
            f"{class_mixed.mean() * 100:.2f}%"
        )


def evaluate_knn(X, y):
    print("\n" + "=" * 60)
    print("           KNN CROSS-VALIDATION")
    print("=" * 60)

    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ))
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv
    )

    print("\nAccuracy:", round(
        accuracy_score(y, predictions), 4
    ))

    print("F1 Macro:", round(
        f1_score(y, predictions, average="macro"), 4
    ))

    print("\nClassification Report:")
    print(classification_report(
        y,
        predictions,
        labels=CLASSES,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y,
        predictions,
        labels=CLASSES
    ))


def run_analysis():
    print("Loading training data...")

    X_train, y_train = prepare_data()

    print("Training samples:", len(X_train))
    print("Features:", len(X_train.columns))

    analyze_neighborhood_purity(X_train, y_train)

    evaluate_knn(X_train, y_train)

    print("\nData separability investigation completed.")


if __name__ == "__main__":
    run_analysis()

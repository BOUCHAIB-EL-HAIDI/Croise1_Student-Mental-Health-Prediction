import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0 , str(PROJECT_ROOT))

import pandas as pd
from src.data.data_loader import load_dataset


TARGET = "Mental_Health_Risk"


def show_overview(df):
    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Columns ---")
    print(df.columns.tolist())

    print("\n--- Features ---")
    print(df.drop(columns=[TARGET, "Student_ID"], errors="ignore").columns.tolist())

    print("\n--- Target ---")
    print(TARGET)

    print("\n--- Data Types ---")
    print(df.dtypes)


def analyze_data_quality(df):
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Missing Values Percentage ---")
    print((df.isnull().mean() * 100).round(2))

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    if "Student_ID" in df.columns:
        print("\n--- Duplicate Student IDs ---")
        print(df["Student_ID"].duplicated().sum())

    print("\n--- Unique Categorical Values ---")
    categorical_columns = df.select_dtypes(include="object").columns.drop(
        "Student_ID", errors="ignore"
    )

    for column in categorical_columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))


def analyze_statistics(df):
    print("\n--- Numerical Statistics ---")
    print(df.describe().round(2))

    print("\n--- Categorical Statistics ---")
    print(df.describe(include="object"))

    print("\n--- Potential Outliers (IQR) ---")

    numerical_columns = df.select_dtypes(include="number").columns.drop(
        "Student_ID", errors="ignore"
    )

    for column in numerical_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df.loc[
            (df[column] < lower_bound) |
            (df[column] > upper_bound),
            column
        ].sort_values()

        print(f"\n{column}")
        print(f"Q1: {Q1:.2f} | Q3: {Q3:.2f}")
        print(f"Lower Bound: {lower_bound:.2f}")
        print(f"Upper Bound: {upper_bound:.2f}")
        print(f"Number of Potential Outliers: {len(outliers)}")

        if not outliers.empty:
            print("Outlier Values:")
            print(outliers.tolist())
        else:
            print("No potential outliers found.")



def analyze_relationships(df):
    numerical_columns = df.select_dtypes(include="number").columns.drop(
        "Student_ID", errors="ignore"
    )

    print("\n--- Target Distribution ---")
    print(df[TARGET].value_counts())

    print("\n--- Target Distribution Percentage ---")
    print(df[TARGET].value_counts(normalize=True).mul(100).round(2))

    print("\n--- Numerical Features vs Target ---")
    print(df.groupby(TARGET)[numerical_columns].mean())

    categorical_columns = df.select_dtypes(include="object").columns.drop(
        [TARGET, "Student_ID"], errors="ignore"
    )

    print("\n--- Categorical Features vs Target ---")
    for column in categorical_columns:
        print(f"\n{column}:")
        print(pd.crosstab(
            df[column],
            df[TARGET],
            normalize="index"
        ).round(2))

    print("\n--- Numerical Correlations ---")
    print(df[numerical_columns].corr())


def analyze_inconsistencies(df):
    print("\n--- Numerical Range Validation ---")

    ranges = {
        "Age": (16, 100),
        "Year_of_Study": (1, 5),
        "Sleep_Hours": (0, 24),
        "Study_Hours_Per_Week": (0, 168),
        "GPA": (0, 20),
        "Social_Media_Hours": (0, 24),
        "Screen_Time_Hours": (0, 24),
        "Financial_Stress": (1, 10),
        "Family_Support": (1, 10),
        "Attendance_Rate": (0, 100)
    }

    for column, (minimum, maximum) in ranges.items():
        if column in df.columns:
            invalid = df[
                (df[column] < minimum) |
                (df[column] > maximum)
            ]

            print(f"{column}: {len(invalid)} invalid values")

    print("\n--- Categorical Validation ---")

    expected_values = {
        "Gender": ["Female", "Male"],
        "Physical_Activity": ["Low", "Moderate", "High"],
        "Part_Time_Job": ["Yes", "No"],
        "Diet_Quality": ["Poor", "Average", "Good"],
        "Previous_Mental_Health_History": ["Yes", "No"],
        "Mental_Health_Risk": ["Low", "Medium", "High"]
    }

    for column, expected in expected_values.items():
        if column in df.columns:
            unexpected = df[
                df[column].notna() &
                ~df[column].isin(expected)
            ][column].unique()

            print(f"{column}: {unexpected}")

    print("\n--- Logical Validation ---")

    if "Year_of_Study" in df.columns:
        invalid_years = df[
            df["Year_of_Study"].notna() &
            (df["Year_of_Study"] % 1 != 0)
        ]

        print("Non-integer study years:", len(invalid_years))

    if "Student_ID" in df.columns:
        print("Missing Student IDs:", df["Student_ID"].isnull().sum())
        print("Duplicate Student IDs:", df["Student_ID"].duplicated().sum())


def profile_dataset(df):
    show_overview(df)
    analyze_data_quality(df)
    analyze_statistics(df)
    analyze_relationships(df)
    analyze_inconsistencies(df)


def main():
    df = load_dataset()
    profile_dataset(df)


if __name__ == "__main__":
    main()
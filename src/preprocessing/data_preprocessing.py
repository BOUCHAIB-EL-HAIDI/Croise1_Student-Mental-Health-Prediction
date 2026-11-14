
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

TARGET = "Mental_Health_Risk"


def split_data(df):
    X = df.drop(columns=[TARGET, "Student_ID"])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def create_preprocessor(X_train):
    numerical_columns = X_train.select_dtypes(include="number").columns

    onehot_columns = ["Gender"]

    binary_columns = [
        "Part_Time_Job",
        "Previous_Mental_Health_History"
    ]

    ordinal_columns = [
        "Physical_Activity",
        "Diet_Quality"
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                ["No", "Yes"],
                ["No", "Yes"]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                ["Low", "Moderate", "High"],
                ["Poor", "Average", "Good"]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("onehot", onehot_pipeline, onehot_columns),
        ("binary", binary_pipeline, binary_columns),
        ("ordinal", ordinal_pipeline, ordinal_columns)
    ])

    return preprocessor

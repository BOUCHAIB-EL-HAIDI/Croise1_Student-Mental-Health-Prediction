
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"


def load_dataset(file_path=DEFAULT_DATA_PATH):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The dataset file is empty.") from error
    except pd.errors.ParserError as error:
        raise ValueError("Unable to parse the CSV file.") from error

    if df.empty:
        raise ValueError("The dataset contains no records.")

    return df


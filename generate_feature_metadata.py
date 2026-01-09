import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# --- Configuration ---
OUTPUT_DIR = Path("assets/actions")
OUTPUT_FILE = OUTPUT_DIR / "feature_metadata.json"

# Map dataset names to file paths
DATASET_PATHS = {
    "adult": "assets/data/s_adult.csv",
    "home": "assets/data/s_home.csv",
    "student": "assets/data/s_student.csv",
    "credit_card": "assets/data/extra/s_credit_cart.csv"
}

# Target columns to exclude from action generation
TARGET_COLS = {
    "adult": "income",
    "home": "RiskPerformance",
    "student": "final_result",
    "credit_card": "Y"
}

# Features users cannot change
IMMUTABLE_CONSTRAINTS = {
    "adult": [
        "age", "race", "sex", "native_country", "marital_status", "relationship"
    ],
    "home": [
        "MSinceOldestTradeOpen"
    ],
    "student": [
        "gender", "region", "disability", "id_student", "imd_band", "age_band"
    ],
    "credit_card": [
        "SEX", "MARRIAGE", "AGE", "ID"
    ]
}


def get_action_metadata(df, dataset_name):
    """
    Analyzes a dataframe and returns a dictionary of actionable features.
    """
    metadata = {}
    target = TARGET_COLS.get(dataset_name, df.columns[-1])
    immutable = IMMUTABLE_CONSTRAINTS.get(dataset_name, [])

    print(f"Processing {dataset_name}...")

    for col in df.columns:
        # Skip target and ID columns
        if col == target or col.lower() in ['id', 'id_student']:
            continue

        is_mutable = col not in immutable

        col_meta = {
            "mutable": is_mutable,
            "name": col
        }

        # Analyze feature types
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check discrete
            if len(df[col].unique()) < 20:
                col_meta["type"] = "discrete"
                col_meta["domain"] = sorted(df[col].dropna().unique().tolist())
                # If mutable and discrete, steps are 1 unit
                col_meta["step_size"] = 1
            else:
                col_meta["type"] = "continuous"
                # Exclude outliers
                col_meta["min"] = float(df[col].quantile(0.01))
                col_meta["max"] = float(df[col].quantile(0.99))
                col_meta["mean"] = float(df[col].mean())

                # Step size 10% of standard deviation
                std_dev = df[col].std()
                col_meta["step_size"] = float(std_dev / 10.0) if std_dev > 0 else 1.0

        else:
            # categorical
            col_meta["type"] = "categorical"
            col_meta["domain"] = df[col].dropna().unique().tolist()

        metadata[col] = col_meta

    return metadata


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_metadata = {}

    for name, path_str in DATASET_PATHS.items():
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Data file not found for {name} at {path}. Skipping.")
            continue

        try:
            # Read CSV
            df = pd.read_csv(path)

            # Generate Metadata
            dataset_meta = get_action_metadata(df, name)
            full_metadata[name] = dataset_meta

        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(full_metadata, f, indent=4)

    print(f"\nSuccess! Feature metadata generated at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
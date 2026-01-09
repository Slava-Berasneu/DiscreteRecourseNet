import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# configuration
CONFIG_DIR = Path("assets/configs")
ACTION_METADATA_PATH = CONFIG_DIR / "feature_metadata.json"
OUTPUT_DIR = Path("assets/actions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_action_metadata():
    if ACTION_METADATA_PATH.exists():
        with open(ACTION_METADATA_PATH, 'r') as f:
            return json.load(f)
    print("Warning: feature_metadata.json not found. Falling back to simple encoding.")
    return {}


def get_dataset_paths():
    """Scans config directory for dataset paths."""
    dataset_map = {}
    search_paths = [CONFIG_DIR, CONFIG_DIR / "extra"]

    for path in search_paths:
        if not path.exists(): continue
        for file in path.glob("*.json"):
            if "action" in file.name or "meta" in file.name: continue
            try:
                with open(file, 'r') as f:
                    config = json.load(f)
                if 'dataset_name' in config and 'data_dir' in config:
                    dataset_map[config['dataset_name']] = config['data_dir']
            except Exception:
                pass
    return dataset_map


def preprocess_with_metadata(df, dataset_name, metadata):
    """
    Encodes columns based on the definitions in action_metadata.json.
    """
    df_encoded = df.copy()
    dataset_meta = metadata.get(dataset_name, {})

    for col in df_encoded.columns:
        col_meta = dataset_meta.get(col, {})
        ctype = col_meta.get("type", "unknown")

        # ordinal variables in domain order
        if ctype == "ordinal" and "domain" in col_meta:
            domain_order = col_meta["domain"]
            # Create a mapping: { "Preschool": 0, "HS-grad": 8, ... }
            mapping = {val: idx for idx, val in enumerate(domain_order)}

            # Map and fill unknown with -1
            df_encoded[col] = df_encoded[col].map(mapping)
            # Fill any values not in the domain with -1
            df_encoded[col] = df_encoded[col].fillna(-1)

        # categorical variables
        elif ctype == "categorical" or df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = le.fit_transform(df_encoded[col])

        # continuous / discrete
        else:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

    return df_encoded


def generate_heatmap(name, csv_path, metadata):
    print(f"Processing {name}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f" [Error] File not found: {csv_path}")
        return

    # preprocessing
    processed_df = preprocess_with_metadata(df, name, metadata)

    # compute correlation (Spearman)
    corr_matrix = processed_df.corr(method='spearman')

    # plotting
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='coolwarm',
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    plt.title(f"Feature Dependency (Spearman) - {name.capitalize()}", fontsize=16)
    plt.tight_layout()

    save_path = OUTPUT_DIR / f"{name}_heatmap.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" Saved heatmap to {save_path}")

    # dependency analysis
    print(f" [Analysis] Potential Dependencies (|r| > 0.7):")
    stack = corr_matrix.abs().stack()
    pairs = stack[stack < 1.0].sort_values(ascending=False)

    seen = set()
    count = 0
    for index, val in pairs.items():
        if val > 0.7:
            a, b = index
            if (b, a) not in seen:
                print(f" - {a} <--> {b}: {val:.2f}")
                seen.add((a, b))
                count += 1
    if count == 0:
        print(" None found.")
    print("-" * 30)


if __name__ == "__main__":
    datasets = get_dataset_paths()
    metadata = load_action_metadata()

    if not datasets:
        print("No dataset configs found.")

    for name, path in datasets.items():
        generate_heatmap(name, path, metadata)
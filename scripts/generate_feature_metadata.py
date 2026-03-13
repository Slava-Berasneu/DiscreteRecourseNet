"""
Generate feature_metadata.json for DiscreteRecourseNet.

Schema:
Each dataset maps feature-name -> metadata.

All features include:
  - "mutable": bool

If mutable == False:
  - output ONLY {"mutable": false}

If mutable == True:
  - include "type" and its fields:

  continuous:
    {"mutable":true, "type":"continuous", "min":..., "max":..., "step_size":...}

  ordinal:
    {"mutable":true, "type":"ordinal", "domain":[...]}

  categorical:
    {"mutable":true, "type":"categorical", "domain":[...]}

Sentinel values:

Some datasets use sentinel codes (e.g. home: -9/-8/-7). We:
  - exclude them from numeric stats (min/max/std),
  - keep them recorded under special_values,
  - for ordinal/categorical domains, we keep them in the domain

This script also writes monotonicity.json:
  {<dataset>: {"increase_only": [...], "decrease_only": [...]}}

Which is used by generate_action_groups_monotonicity.py to add
monotonicity constraints into action_groups.json

"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATHS = {
    "adult": PROJECT_ROOT / "assets" / "data" / "s_adult.csv",
    "home": PROJECT_ROOT / "assets" / "data" / "s_home.csv",
    "student": PROJECT_ROOT / "assets" / "data" / "s_student.csv",
    "credit_card": PROJECT_ROOT / "assets" / "data" / "extra" / "s_credit_cart.csv",
}

TARGET_COLS = {
    "adult": "income",
    "home": "RiskPerformance",
    "student": "final_result",
    "credit_card": "Y",
}

IMMUTABLE_CONSTRAINTS = {
    "adult": ["age", "race", "gender", "marital_status"],
    "home": ["MSinceOldestTradeOpen", "ExternalRiskEstimate"],
    "student": [
        "gender",
        "region",
        "disability",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "weight",
        "code_module",
    ],
    "credit_card": ["SEX", "MARRIAGE", "EDUCATION", "AGE"],
}

# Force a feature to be interpreted as the given kind
FORCE_ORDINAL_FEATURES = {
    "student": ["studied_credits"],
}

FORCE_CATEGORICAL_FEATURES = {
    "adult": ["education"],
}

# Define orderings for ordinal string features
ORDINAL_DOMAIN_OVERRIDES: Dict[str, Dict[str, List[Any]]] = {
    "student": {
        "highest_education": [
            "No Formal quals",
            "Lower Than A Level",
            "A Level or Equivalent",
            "HE Qualification",
            "Post Graduate Qualification",
        ],
    },
    "home": {
        "MaxDelq2PublicRecLast12M": [
            "Never Delinquent",
            "30 Days Delinquent",
            "60 Days Delinquent",
            "90 Days Delinquent",
            "120+ Days Delinquent",
            "Derogatory Comment",
        ],
        "MaxDelqEver": [
            "Never Delinquent",
            "30 Days Delinquent",
            "60 Days Delinquent",
            "90 Days Delinquent",
            "120+ Days Delinquent",
            "Derogatory Comment",
        ],
    },
    "credit_card": {
        "PAY_0": list(range(-2, 9)),
        "PAY_2": list(range(-2, 9)),
        "PAY_3": list(range(-2, 9)),
        "PAY_4": list(range(-2, 9)),
        "PAY_5": list(range(-2, 9)),
        "PAY_6": list(range(-2, 9)),
    },
}

ORDINAL_UNIQUE_THRESHOLD = 20
MAX_ORDINAL_DOMAIN_SIZE = 20000

# Special values
SPECIAL_VALUES = {
    "home": {
        "global": [-9, -8, -7],
        "per_feature": {},
    }
}

# Monotonicity constraints
MONOTONICITY_BY_DATASET = {
    "adult": {"increase_only": [], "decrease_only": []},
    "home": {
        "increase_only": [
            "PercentTradesNeverDelq",
            "MSinceMostRecentDelq",
            "MSinceMostRecentInqexcl7days",
            "NumSatisfactoryTrades",
            "NumTotalTrades",
            "NumTradesOpeninLast12M",
            "NumTrades60Ever2DerogPubRec",
            "NumTrades90Ever2DerogPubRec",
        ],
        "decrease_only": [
            "NumInqLast6M",
            "NumInqLast6Mexcl7days",
            "NetFractionRevolvingBurden",
            "NetFractionInstallBurden",
            "NumRevolvingTradesWBalance",
            "NumInstallTradesWBalance",
            "NumBank2NatlTradesWHighUtilization",
            "PercentTradesWBalance",
        ],
    },
    "student": {
        "increase_only": [
            "weighted_score",
            "studied_credits",
            "forumng_click",
            "homepage_click",
            "oucontent_click",
            "resource_click",
            "subpage_click",
            "url_click",
            "dataplus_click",
            "glossary_click",
            "oucollaborate_click",
            "quiz_click",
            "ouelluminate_click",
            "sharedsubpage_click",
            "questionnaire_click",
            "page_click",
            "externalquiz_click",
            "ouwiki_click",
            "dualpane_click",
            "folder_click",
            "repeatactivity_click",
            "htmlactivity_click",
            "highest_education",
        ],
        "decrease_only": [],
    },
    "credit_card": {
        "increase_only": [
            "LIMIT_BAL",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ],
        "decrease_only": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    },
}


def _round_sig(x: float, sig: int = 3) -> float:
    if x == 0 or not np.isfinite(x):
        return float(x)
    return float(round(x, sig - int(math.floor(math.log10(abs(x)))) - 1))


def _is_integer_like(series: pd.Series, tol: float = 1e-6) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    vals = series.dropna().to_numpy()
    if vals.size == 0:
        return False
    return bool(np.all(np.isclose(vals, np.round(vals), atol=tol, rtol=0.0)))


def _step_from_std(series: pd.Series) -> float:
    std = float(series.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        return 1.0
    return float(std / 10.0)


def _as_sorted_domain(values: List[Any]) -> List[Any]:
    try:
        return sorted(values)
    except TypeError:
        return sorted([str(v) for v in values])


def _get_special_values_present(series: pd.Series, dataset_name: str, feature_name: str) -> List[Any]:
    cfg = SPECIAL_VALUES.get(dataset_name)
    if not cfg:
        return []
    specials = set(cfg.get("global", [])) | set(cfg.get("per_feature", {}).get(feature_name, []))
    if not specials:
        return []

    if pd.api.types.is_numeric_dtype(series):
        numeric_specials = [v for v in specials if isinstance(v, (int, float, np.integer, np.floating))]
        present = [v for v in numeric_specials if series.isin([v]).any()]
        # normalize floats that are ints
        out = []
        for v in sorted(set(present), key=float):
            if isinstance(v, float) and float(v).is_integer():
                out.append(int(v))
            else:
                out.append(v)
        return out

    s_str = series.astype(str)
    present = [str(v) for v in specials if s_str.isin([str(v)]).any()]
    return _as_sorted_domain(list(set(present)))


def _clean_for_stats(series: pd.Series, specials_present: List[Any]) -> pd.Series:
    if not specials_present:
        return series
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        s = s.mask(s.isin(specials_present), np.nan)
    else:
        sv = set(str(v) for v in specials_present)
        s_str = s.astype(str)
        s = s.mask(s_str.isin(sv), np.nan)
    return s


def _merge_specials_into_domain(domain: List[Any], specials_present: List[Any]) -> List[Any]:
    if not specials_present:
        return domain
    # numeric
    if all(isinstance(x, (int, float, np.integer, np.floating)) for x in specials_present) and \
       all(isinstance(x, (int, float, np.integer, np.floating)) for x in domain):
        merged = sorted(set(domain) | set(specials_present), key=float)
        out = []
        for v in merged:
            if isinstance(v, float) and float(v).is_integer():
                out.append(int(v))
            else:
                out.append(v)
        return out

    # strings
    dom_str = [str(x) for x in domain]
    specials_str = _as_sorted_domain([str(x) for x in specials_present if str(x) not in dom_str])
    return specials_str + domain


def _infer_numeric_ordinal_domain(series_stats: pd.Series) -> List[Any]:
    vals = series_stats.dropna().to_numpy()
    if vals.size == 0:
        return []
    uniq = np.unique(vals)

    if np.all(np.isclose(uniq, np.round(uniq))):
        uniq_int = np.sort(np.unique(np.round(uniq).astype(int)))
        diffs = np.diff(uniq_int)
        diffs = diffs[diffs != 0]
        step = 1
        if diffs.size:
            step = int(np.gcd.reduce(np.abs(diffs)))
            step = max(step, 1)
        full = np.arange(uniq_int.min(), uniq_int.max() + step, step)
        if full.size <= MAX_ORDINAL_DOMAIN_SIZE:
            return full.tolist()
        return uniq_int.tolist()

    uniq_sorted = np.sort(uniq)
    return uniq_sorted.tolist()


def _resolve_dataset_path(dataset_name: str, path_str: Union[str, Path]) -> Path:
    p = Path(path_str)
    if p.exists():
        return p

    if not p.is_absolute():
        project_candidate = PROJECT_ROOT / p
        if project_candidate.exists():
            return project_candidate

    data_dir = PROJECT_ROOT / "assets" / "data"
    if dataset_name == "credit_card":
        candidates = [data_dir / "extra" / "s_credit_cart.csv", data_dir / "extra" / "s_credit_card.csv"]
    else:
        candidates = [data_dir / f"s_{dataset_name}.csv"]
    for c in candidates:
        if c.exists():
            return c

    return p


def get_feature_metadata(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    target = TARGET_COLS.get(dataset_name, df.columns[-1])
    immutable = set(IMMUTABLE_CONSTRAINTS.get(dataset_name, []))
    forced_ordinal = set(FORCE_ORDINAL_FEATURES.get(dataset_name, []))
    forced_categorical = set(FORCE_CATEGORICAL_FEATURES.get(dataset_name, []))

    meta: Dict[str, Any] = {}

    for col in df.columns:
        if col == target:
            continue

        if col in immutable:
            meta[col] = {"mutable": False}
            continue

        s_raw = df[col]
        specials_present = _get_special_values_present(s_raw, dataset_name, col)
        s_stats = _clean_for_stats(s_raw, specials_present)

        nunique_real = int(s_stats.dropna().nunique())

        # categorical
        if col in forced_categorical:
            domain = _as_sorted_domain([str(x) for x in s_raw.dropna().unique().tolist()])
            domain = _merge_specials_into_domain(domain, specials_present)
            entry: Dict[str, Any] = {"mutable": True, "type": "categorical", "domain": domain}
            if specials_present:
                entry["special_values"] = specials_present
            meta[col] = entry
            continue

        # ordinal
        if col in ORDINAL_DOMAIN_OVERRIDES.get(dataset_name, {}):
            base_domain = ORDINAL_DOMAIN_OVERRIDES[dataset_name][col]
            # preserve special values
            observed = s_raw.dropna().unique().tolist()
            extras = [v for v in _as_sorted_domain(observed) if v not in base_domain]
            domain = list(base_domain) + [v for v in extras if v not in base_domain]
            domain = _merge_specials_into_domain(domain, specials_present)
            entry = {"mutable": True, "type": "ordinal", "domain": domain}
            if specials_present:
                entry["special_values"] = specials_present
            meta[col] = entry
            continue

        # ordinal numeric
        if col in forced_ordinal and pd.api.types.is_numeric_dtype(s_raw):
            dom = _infer_numeric_ordinal_domain(s_stats)
            # normalize ints
            dom2 = []
            for v in dom:
                if isinstance(v, (np.integer, int)):
                    dom2.append(int(v))
                elif isinstance(v, (np.floating, float)) and float(v).is_integer():
                    dom2.append(int(float(v)))
                else:
                    dom2.append(v)
            dom2 = _merge_specials_into_domain(dom2, specials_present)
            entry = {"mutable": True, "type": "ordinal", "domain": dom2}
            if specials_present:
                entry["special_values"] = specials_present
            meta[col] = entry
            continue

        # numeric
        if pd.api.types.is_numeric_dtype(s_raw):
            if nunique_real <= ORDINAL_UNIQUE_THRESHOLD:
                dom = _infer_numeric_ordinal_domain(s_stats)
                dom2 = []
                for v in dom:
                    if isinstance(v, (np.integer, int)):
                        dom2.append(int(v))
                    elif isinstance(v, (np.floating, float)) and float(v).is_integer():
                        dom2.append(int(float(v)))
                    else:
                        dom2.append(v)
                dom2 = _merge_specials_into_domain(dom2, specials_present)
                entry = {"mutable": True, "type": "ordinal", "domain": dom2}
                if specials_present:
                    entry["special_values"] = specials_present
                meta[col] = entry
            else:
                if s_stats.dropna().shape[0] == 0:
                    domain = _as_sorted_domain(s_raw.dropna().unique().tolist())
                    entry = {"mutable": True, "type": "categorical", "domain": domain}
                    if specials_present:
                        entry["special_values"] = specials_present
                    meta[col] = entry
                else:
                    int_like = _is_integer_like(s_stats)
                    mn = float(s_stats.min())
                    mx = float(s_stats.max())
                    step = _step_from_std(s_stats)

                    if int_like:
                        mn = int(round(mn))
                        mx = int(round(mx))
                        step = max(1, int(round(step)))
                    else:
                        step = _round_sig(step, sig=3)

                    entry = {
                        "mutable": True,
                        "type": "continuous",
                        "min": _round_sig(mn, sig=3),
                        "max": _round_sig(mx, sig=3),
                        "step_size": step,
                    }
                    if specials_present:
                        entry["special_values"] = specials_present
                    meta[col] = entry
        else:
            domain = _as_sorted_domain([str(x) for x in s_raw.dropna().unique().tolist()])
            domain = _merge_specials_into_domain(domain, specials_present)
            entry = {"mutable": True, "type": "categorical", "domain": domain}
            if specials_present:
                entry["special_values"] = specials_present
            meta[col] = entry

    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "assets" / "actions" / "feature_metadata.json"),
        help="Output path",
    )
    p.add_argument(
        "--monotonicity-out",
        type=str,
        default=str(PROJECT_ROOT / "assets" / "actions" / "monotonicity.json"),
        help="Monotonicity sidecar output",
    )
    p.add_argument("--datasets", type=str, nargs="*", default=list(DEFAULT_DATASET_PATHS.keys()), help="Datasets to process")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    full: Dict[str, Any] = {}
    for name in args.datasets:
        path = _resolve_dataset_path(name, DEFAULT_DATASET_PATHS.get(name, ""))
        if not path.exists():
            print(f"Warning: data file not found for {name} at {path}. Skipping.")
            continue
        df = pd.read_csv(path)
        full[name] = get_feature_metadata(df, name)

    out_path.write_text(json.dumps(full, indent=4), encoding="utf-8")
    print(f"Success! Feature metadata generated at: {out_path}")

    mono_path = Path(args.monotonicity_out)
    mono_path.parent.mkdir(parents=True, exist_ok=True)
    mono_path.write_text(json.dumps(MONOTONICITY_BY_DATASET, indent=4), encoding="utf-8")
    print(f"Success! Monotonicity information written to: {mono_path}")


if __name__ == "__main__":
    main()

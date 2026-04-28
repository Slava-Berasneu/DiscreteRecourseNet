from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PREFIX_RX = re.compile(
    r"^(?P<tag>.+?)__(?P<gs>gs\d+)__(?P<params>.+)$"
)
_KV_RX = re.compile(r"(?P<key>[a-z][a-z_0-9]*)-(?P<val>[^_]+)")


def tag_to_float(tag: Optional[str]) -> Optional[float]:
    if tag is None:
        return None
    return float(tag.replace("p", "."))


def _parse_csv_floats(s: Optional[str]) -> Optional[Set[float]]:
    if s is None:
        return None
    return {float(x) for x in s.split(",") if x.strip()}


def parse_run_dir(run_dir: Path) -> dict:
    m = _PREFIX_RX.match(run_dir.name)
    if not m:
        raise ValueError(f"Unrecognized run directory name: {run_dir.name}")

    result = {"run_dir": run_dir.name, "tag": m["tag"], "gs": m["gs"]}
    for kv in _KV_RX.finditer(m["params"]):
        result[kv["key"]] = tag_to_float(kv["val"])
    return result


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path, index_col=0)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={df.index.name or "index": "model"})
    if "index" in df.columns and "model" not in df.columns:
        df = df.rename(columns={"index": "model"})
    return df


def build_table(
    results_root: Path,
    dataset: str,
    seed: int,
    include_old_names: bool,
    *,
    filter_lambda2: Optional[Set[float]] = None,
    filter_action_cost_base: Optional[Set[float]] = None,
    filter_tau_mask: Optional[Set[float]] = None,
    filter_run_tag: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir() or not _PREFIX_RX.match(run_dir.name):
            continue

        meta = parse_run_dir(run_dir)

        if filter_run_tag is not None and meta.get("tag") != filter_run_tag:
            continue
        if not include_old_names and meta.get("l2_l3_target_ratio") is not None:
            continue

        if filter_lambda2 is not None and meta.get("lambda_2") not in filter_lambda2:
            continue
        if filter_action_cost_base is not None and meta.get("action_cost_base") not in filter_action_cost_base:
            continue
        if filter_tau_mask is not None and meta.get("gumbel_tau_mask") not in filter_tau_mask:
            continue

        metrics_path = run_dir / dataset / f"seed-{seed}" / "metrics.csv"
        if not metrics_path.exists():
            continue

        metrics_df = load_metrics(metrics_path)
        if metrics_df.empty:
            continue

        for _, metric_row in metrics_df.iterrows():
            row = {**meta, **metric_row.to_dict()}
            row["metrics_path"] = str(metrics_path)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["lambda_2", "action_cost_base", "gumbel_tau_mask", "gumbel_tau_choice", "model"] if c in out.columns]
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile grid-search metrics into one table.")
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "assets" / "results"),
        help="Root results directory.",
    )
    parser.add_argument("--dataset", default="home", help="Dataset subdirectory name.")
    parser.add_argument("--seed", type=int, default=31, help="Seed number to read.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <results-root>/compiled_<dataset>_seed-<seed>_metrics.csv",
    )
    parser.add_argument(
        "--include-old-names",
        action="store_true",
        help="Also include folders that still contain l2_l3_target_ratio in the run name.",
    )

    parser.add_argument(
        "--filter-lambda2",
        type=str,
        default=None,
        help="Comma-separated lambda_2 values to include (e.g. '0.0005,0.001').",
    )
    parser.add_argument(
        "--filter-action-cost-base",
        type=str,
        default=None,
        help="Comma-separated action_cost_base values to include.",
    )
    parser.add_argument(
        "--filter-tau-mask",
        type=str,
        default=None,
        help="Comma-separated gumbel_tau_mask values to include (e.g. '0.5,0.75,1.0,1.5').",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Only include runs whose tag prefix matches this value (e.g. 'home_gs3').",
    )

    args = parser.parse_args()

    results_root = Path(args.results_root)
    output = Path(args.output) if args.output else results_root / f"compiled_{args.dataset}_seed-{args.seed}_metrics.csv"

    table = build_table(
        results_root,
        args.dataset,
        args.seed,
        args.include_old_names,
        filter_lambda2=_parse_csv_floats(args.filter_lambda2),
        filter_action_cost_base=_parse_csv_floats(args.filter_action_cost_base),
        filter_tau_mask=_parse_csv_floats(args.filter_tau_mask),
        filter_run_tag=args.run_tag,
    )
    if table.empty:
        raise SystemExit("No matching metrics.csv files found.")

    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(table.to_string(index=False))
    print(f"\nWrote {len(table)} rows to {output}")


if __name__ == "__main__":
    main()

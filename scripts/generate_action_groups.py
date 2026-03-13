"""
Generate `action_groups.json` from `feature_metadata.json` and group definitions in this script

The generated `action_groups.json` allows the model to:
  (1) select action groups and discrete interventions, and
  (2) perform invalid action masking

Monotonicity gets added as

  <dataset>.__constraints__.monotonicity = {
      "increase_only": [...],
      "decrease_only": [...]
  }

Features not listed are treated as bidirectional.

Output structure:
{
  "adult": {
    "__constraints__": {"monotonicity": {"increase_only": [...], "decrease_only": [...]}},
    "<group_id>": { ...group object... },
    ...
  },
  "home": {...},
  ...
}

Action group types:

Type 0: Independent singleton group (one feature)
Type 1: Physically constrained (base -> derived via deterministic map)
Type 2: Part-whole (base -> derived via deterministic map)
Type 3: Momentum / temporal sequence
Type 4: Latent behavior cluster
Type 5: One-hot choice block

Action domains (finite):

- kind="noop"        : {"kind":"noop","values":[0]} (immutable)
- kind="values"      : {"kind":"values","feature":"f","values":[...]} (single head)
- kind="cartesian"   : {"kind":"cartesian","features":[...],"domains":{f:[...],...}} (multi-head)
- kind="delta_steps" : {"kind":"delta_steps","deltas":[...],"scales":{f:s,...}} (Type 4)
- kind="choice"      : {"kind":"choice","values":[1..m],"mapping":{...}} (Type 5)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

Number = Union[int, float]
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add/modify groups here. Anything not listed becomes Type 0
'''
GROUP_SPECS: Dict[str, List[Dict[str, Any]]] = {
    "adult": [],
    "home": [
        {
            "id": "Stop_New_Credit_Applications",
            "type": 2,
            "name": "Stop new credit applications",
            "description": "Part-whole (subset). Stopping new applications reduces recent inquiry counts.",
            "roles": {"base": ["NumInqLast6M"], "derived": ["NumInqLast6Mexcl7days"]},
            "rule": {"kind": "clip_derived_to_base", "params": {"base": "NumInqLast6M"}},
        },
        {
            "id": "Build_Satisfactory_Credit_History",
            "type": 2,
            "name": "Build satisfactory credit history",
            "description": "Part-whole (sum). Accumulating more satisfactory trades increases total trade count.",
            "roles": {"base": ["NumSatisfactoryTrades"], "derived": ["NumTotalTrades"]},
            "rule": {
                "kind": "derived_adds_base_delta",
                "params": {"base": "NumSatisfactoryTrades", "derived": "NumTotalTrades"},
            },
        },
        {
            "id": "Pay_Down_Revolving_Debt",
            "type": 4,
            "name": "Pay down revolving debt",
            "description": "Latent behavior. Paying down balances reduces utilization and high-utilization trade counts.",
            "features": ["NetFractionRevolvingBurden", "NumBank2NatlTradesWHighUtilization"],
            "rule": {"kind": "latent_shift", "params": {"scaling": "step_size"}},
        },
    ],
    "student": [
        {
            "id": "Increase_Online_Participation",
            "type": 4,
            "name": "Increase online participation",
            "description": "Latent behavior. Active participation increases interaction counts for forums/content/homepage.",
            "features": ["forumng_click", "homepage_click", "subpage_click", "resource_click", "url_click", "oucontent_click"],
            "rule": {"kind": "latent_shift", "params": {"scaling": "step_size"}},
        }
    ],
    "credit_card": [
        {
            "id": "Reduce_Spending_Habits",
            "type": 3,
            "name": "Reduce spending habits",
            "description": "Momentum. Reducing spending habits lowers bill amounts sequentially over time.",
            "roles": {"time_order": ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]},
            "rule": {"kind": "exponential_smoothing", "params": {"alpha": 0.7}},
        },
        {
            "id": "Improve_Payment_Consistency",
            "type": 3,
            "name": "Improve payment consistency",
            "description": "Momentum. Paying on time improves the trajectory of payment status history.",
            "roles": {"time_order": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]},
            "rule": {"kind": "sequential_consistency", "params": {}},
        },
    ],
}
'''
GROUP_SPECS: Dict[str, List[Dict[str, Any]]] = {
    "adult": [],
    "home": [],
    "student": [],
    "credit_card": []
}

# Default delta domain for Type 4 groups
DEFAULT_TYPE4_DELTA_DOMAIN: List[int] = [-3, -2, -1, 0, 1, 2, 3]

DEFAULT_MAX_VALUE_LIST_SIZE = 20000
DEFAULT_LIST_WRAP = 20

MONOTONICITY_DEFAULT: Dict[str, Dict[str, List[str]]] = {
    "adult": {
        "increase_only": [],
        "decrease_only": [],
    },
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
        "decrease_only": [
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
        ],
    },
}

def _is_int_like(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _round_float(x: float, ndigits: int = 6) -> float:
    return float(round(x, ndigits))


def _linspace_by_step(mn: Number, mx: Number, step: Number, max_size: int) -> List[Number]:
    if step <= 0:
        raise ValueError(f"step_size must be > 0, got {step}")
    span = mx - mn
    n = int(math.floor(span / step)) + 1
    if n > max_size:
        raise ValueError(
            f"Value list too large: n={n} (> {max_size}). "
            f"Range [{mn}, {mx}] with step={step}. Increase --max-value-list-size or adjust step_size."
        )

    # Integer
    if _is_int_like(mn) and _is_int_like(mx) and _is_int_like(step):
        return [int(mn + i * step) for i in range(n)]

    # Float
    vals: List[Number] = []
    for i in range(n):
        v = mn + i * step
        vals.append(_round_float(float(v), ndigits=6))
    if vals:
        vals[-1] = _round_float(min(float(vals[-1]), float(mx)), ndigits=6)
    return vals


def _infer_numeric_step_from_domain(domain: Sequence[Any]) -> float:
    """Infer a numeric step from a numeric ordinal domain"""
    nums = [v for v in domain if _is_number(v)]
    if len(nums) < 2:
        return 1.0
    nums_sorted = sorted(float(v) for v in nums)
    diffs = [nums_sorted[i + 1] - nums_sorted[i] for i in range(len(nums_sorted) - 1)]
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        return 1.0

    if all(abs(d - round(d)) < 1e-9 for d in diffs):
        g = 0
        for d in diffs:
            g = math.gcd(g, int(round(d)))
        return float(g if g > 0 else 1)

    diffs_sorted = sorted(diffs)
    mid = len(diffs_sorted) // 2
    return float(diffs_sorted[mid])


def _feature_action_values(feature_meta: Dict[str, Any], max_size: int) -> Tuple[List[Any], Optional[List[Any]]]:
    """Return (action_values, special_values).

    action_values: actionable targets (excludes sentinel values)
    special_values: sentinel codes
    """
    if feature_meta.get("mutable") is False:
        return [0], None  # noop

    ftype = feature_meta.get("type")
    special = feature_meta.get("special_values")

    if ftype in ("categorical", "ordinal"):
        dom = list(feature_meta.get("domain", []))
        if special is not None:
            special_set = set(special)
            dom = [v for v in dom if v not in special_set]
        return dom, special

    if ftype == "continuous":
        mn = feature_meta["min"]
        mx = feature_meta["max"]
        step = feature_meta["step_size"]
        vals = _linspace_by_step(mn, mx, step, max_size=max_size)
        if special is not None:
            special_set = set(special)
            vals = [v for v in vals if v not in special_set]
        return vals, special

    raise ValueError(f"Unknown/unsupported feature type: {ftype}")


def _type4_scales(features_meta: Dict[str, Dict[str, Any]], feats: Sequence[str]) -> Dict[str, float]:
    """Compute per-feature scale for Type 4 groups

    scale = step_size for continuous features
    else if numeric ordinal: infer step from domain
    """
    scales: Dict[str, float] = {}
    for f in feats:
        meta = features_meta[f]
        if meta.get("mutable") is False:
            raise ValueError(f"Type 4 group contains immutable feature '{f}' (not allowed).")

        ftype = meta.get("type")
        if ftype == "continuous":
            step = meta.get("step_size")
            if step is None or not _is_number(step) or float(step) <= 0:
                scales[f] = 1.0
            else:
                scales[f] = float(step)
        elif ftype == "ordinal":
            dom = meta.get("domain", [])
            if not dom:
                scales[f] = 1.0
                continue
            # allow numeric ordinal only
            if not all(_is_number(v) for v in dom):
                raise ValueError(f"Type 4 feature '{f}' is non-numeric ordinal; cannot apply additive delta.")
            scales[f] = float(_infer_numeric_step_from_domain(dom))
        else:
            raise ValueError(f"Type 4 feature '{f}' has type '{ftype}' which is not additive.")
    return scales


def _is_scalar(x: Any) -> bool:
    return not isinstance(x, (dict, list))


def _is_scalar_list(xs: list) -> bool:
    return all(_is_scalar(x) for x in xs)


def _json_scalar(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def dumps_pretty(obj: Any, *, indent: int = 2, list_wrap: int = 20) -> str:
    """JSON printer for wrapping long lists"""

    def fmt(o: Any, level: int) -> str:
        sp = " " * (indent * level)

        if isinstance(o, dict):
            if not o:
                return "{}"
            inner = []
            for k, v in o.items():
                key = _json_scalar(k)
                val = fmt(v, level + 1)
                inner.append(f'{" " * (indent * (level + 1))}{key}: {val}')
            return "{\n" + ",\n".join(inner) + "\n" + sp + "}"

        if isinstance(o, list):
            if not o:
                return "[]"
            if _is_scalar_list(o):
                scalars = [_json_scalar(x) for x in o]
                if len(scalars) <= list_wrap:
                    return "[" + ", ".join(scalars) + "]"
                lines = []
                for i in range(0, len(scalars), list_wrap):
                    chunk = ", ".join(scalars[i : i + list_wrap])
                    lines.append(f'{" " * (indent * (level + 1))}{chunk}')
                return "[\n" + ",\n".join(lines) + "\n" + sp + "]"

            inner = []
            for el in o:
                inner.append(f'{" " * (indent * (level + 1))}{fmt(el, level + 1)}')
            return "[\n" + ",\n".join(inner) + "\n" + sp + "]"

        return _json_scalar(o)

    return fmt(obj, 0) + "\n"


def _load_monotonicity(path: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
    """Load monotonicity mapping from JSON"""
    if not path:
        return MONOTONICITY_DEFAULT
    p = Path(path)
    if not p.exists():
        return MONOTONICITY_DEFAULT
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # Basic shape validation
    out: Dict[str, Dict[str, List[str]]] = {}
    for ds, v in obj.items():
        inc = list(v.get("increase_only", []))
        dec = list(v.get("decrease_only", []))
        out[ds] = {"increase_only": inc, "decrease_only": dec}
    return out


def _filter_monotonicity_for_dataset(
    dataset: str,
    features_meta: Dict[str, Dict[str, Any]],
    mono: Dict[str, Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    """Filter monotonicity lists to existing, mutable features."""
    inc = set(mono.get(dataset, {}).get("increase_only", []))
    dec = set(mono.get(dataset, {}).get("decrease_only", []))

    # If a feature is in both, treat as bidirectional
    both = inc & dec
    inc -= both
    dec -= both

    existing = set(features_meta.keys())
    inc = {f for f in inc if f in existing and features_meta[f].get("mutable") is not False}
    dec = {f for f in dec if f in existing and features_meta[f].get("mutable") is not False}

    return {"increase_only": sorted(inc), "decrease_only": sorted(dec)}


def compile_groups_for_dataset(
    dataset: str,
    features_meta: Dict[str, Dict[str, Any]],
    group_specs: Sequence[Dict[str, Any]],
    max_value_list_size: int,
) -> Dict[str, Dict[str, Any]]:
    """Build groups for one dataset.

    - user-defined groups first
    - then Type 0 groups for all remaining features
    """

    groups: Dict[str, Dict[str, Any]] = {}
    used: set[str] = set()

    def add_group(group_id: str, obj: Dict[str, Any]) -> None:
        if group_id in groups:
            raise ValueError(f"Duplicate group id '{group_id}' in dataset '{dataset}'.")
        groups[group_id] = obj

    # user-defined groups
    for spec in group_specs:
        gid = spec["id"]
        gtype = int(spec["type"])
        name = spec.get("name", gid.replace("_", " "))
        desc = spec.get("description", "")

        if gtype not in {0, 1, 2, 3, 4, 5}:
            raise ValueError(f"Invalid group type {gtype} for group '{gid}' in dataset '{dataset}'.")

        roles = spec.get("roles")

        if gtype in {1, 2}:
            roles = spec.get("roles", {})
            base = roles.get("base", [])
            derived = roles.get("derived", [])
            feats = list(base) + list(derived)
        elif gtype == 3:
            roles = spec.get("roles", {})
            feats = list(roles.get("time_order", []))
        elif gtype == 5:
            feats = list(spec.get("features", []))
            roles = spec.get("roles") or {"onehot_block": feats}
        else:
            feats = list(spec.get("features", []))

        if not feats:
            raise ValueError(f"Group '{gid}' in dataset '{dataset}' has no features.")

        # validate features
        for f in feats:
            if f not in features_meta:
                raise ValueError(f"Group '{gid}' references unknown feature '{f}' in dataset '{dataset}'.")
            if f in used:
                raise ValueError(f"Feature '{f}' appears in multiple groups in dataset '{dataset}'.")
            if features_meta[f].get("mutable") is False and len(feats) > 1:
                raise ValueError(
                    f"Immutable feature '{f}' cannot be placed in multi-feature group '{gid}' (dataset '{dataset}')."
                )

        used.update(feats)

        group_mutable = all(features_meta[f].get("mutable") is not False for f in feats)

        # build action_domain by type
        if not group_mutable:
            action_domain: Dict[str, Any] = {"kind": "noop", "values": [0]}
        else:
            if gtype == 0:
                if len(feats) != 1:
                    raise ValueError(f"Type 0 group '{gid}' must be singleton, got {feats}")
                f = feats[0]
                vals, _ = _feature_action_values(features_meta[f], max_size=max_value_list_size)
                action_domain = {"kind": "values", "feature": f, "values": vals}

            elif gtype in {1, 2}:
                base = list(spec["roles"]["base"])
                if not base:
                    raise ValueError(f"Type {gtype} group '{gid}' must define roles.base")
                if len(base) == 1:
                    f0 = base[0]
                    vals, _ = _feature_action_values(features_meta[f0], max_size=max_value_list_size)
                    action_domain = {"kind": "values", "feature": f0, "values": vals}
                else:
                    domains: Dict[str, Any] = {}
                    for f in base:
                        vals, _ = _feature_action_values(features_meta[f], max_size=max_value_list_size)
                        domains[f] = vals
                    action_domain = {"kind": "cartesian", "features": base, "domains": domains}

            elif gtype == 3:
                time_order = list(spec["roles"]["time_order"])
                if not time_order:
                    raise ValueError(f"Type 3 group '{gid}' must define roles.time_order")
                f0 = time_order[0]
                vals, _ = _feature_action_values(features_meta[f0], max_size=max_value_list_size)
                action_domain = {"kind": "values", "feature": f0, "values": vals}

            elif gtype == 4:
                delta_domain = list(spec.get("delta_domain", DEFAULT_TYPE4_DELTA_DOMAIN))
                if not all(isinstance(d, int) for d in delta_domain):
                    raise ValueError(f"Type 4 delta_domain must be integers (delta steps), got {delta_domain}")
                scales = _type4_scales(features_meta, feats)
                action_domain = {"kind": "delta_steps", "deltas": delta_domain, "scales": scales}

            elif gtype == 5:
                m = len(feats)
                action_domain = {
                    "kind": "choice",
                    "values": list(range(1, m + 1)),
                    "mapping": {str(i + 1): feats[i] for i in range(m)},
                }

            else:
                raise ValueError(f"Unhandled group type {gtype} for group '{gid}'")

        # Attach sentinel values from feature_metadata
        special_values_payload: Optional[Any] = None
        if len(feats) == 1:
            sv = features_meta[feats[0]].get("special_values")
            if sv is not None and len(sv) > 0:
                special_values_payload = list(sv)
        else:
            sv_map = {f: features_meta[f].get("special_values") for f in feats if features_meta[f].get("special_values")}
            if len(sv_map) > 0:
                special_values_payload = sv_map

        group_obj: Dict[str, Any] = {
            "type": gtype,
            "name": name,
            "description": desc,
            "features": feats,
            "mutable": bool(group_mutable),
            "action_domain": action_domain,
        }

        if special_values_payload is not None:
            group_obj["special_values"] = special_values_payload

        if roles is not None and gtype in {1, 2, 3, 5}:
            group_obj["roles"] = roles

        if "rule" in spec and spec["rule"] is not None:
            group_obj["rule"] = spec["rule"]

        add_group(gid, group_obj)

    # add remaining features as Type 0 groups
    for f, meta in features_meta.items():
        if f in used:
            continue
        gid = f
        group_mutable = meta.get("mutable") is not False

        if group_mutable:
            vals, _ = _feature_action_values(meta, max_size=max_value_list_size)
            action_domain = {"kind": "values", "feature": f, "values": vals}
        else:
            action_domain = {"kind": "noop", "values": [0]}
        group_obj2: Dict[str, Any] = {
            "type": 0,
            "name": f,
            "description": "Independent feature",
            "features": [f],
            "mutable": bool(group_mutable),
            "action_domain": action_domain,
        }
        sv2 = meta.get("special_values")
        if sv2 is not None and len(sv2) > 0:
            group_obj2["special_values"] = list(sv2)
        add_group(gid, group_obj2)

    return groups


def generate_action_groups(
    feature_metadata: Dict[str, Dict[str, Dict[str, Any]]],
    monotonicity: Dict[str, Dict[str, List[str]]],
    max_value_list_size: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for dataset, feats_meta in feature_metadata.items():
        specs = GROUP_SPECS.get(dataset, [])
        groups = compile_groups_for_dataset(
            dataset=dataset,
            features_meta=feats_meta,
            group_specs=specs,
            max_value_list_size=max_value_list_size,
        )

        ds_mono = _filter_monotonicity_for_dataset(dataset, feats_meta, monotonicity)

        # constraints reserved key
        ds_out: Dict[str, Any] = {"__constraints__": {"monotonicity": ds_mono}}
        ds_out.update(groups)
        out[dataset] = ds_out

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--feature-metadata",
        type=str,
        default=str(PROJECT_ROOT / "assets" / "actions" / "feature_metadata.json"),
        help="Path to feature_metadata.json",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "assets" / "actions" / "action_groups.json"),
        help="Output path for action_groups.json",
    )
    p.add_argument(
        "--monotonicity",
        type=str,
        default=str(PROJECT_ROOT / "assets" / "actions" / "monotonicity.json"),
        help="Optional path to monotonicity.json (overrides defaults if exists).",
    )
    p.add_argument(
        "--max-value-list-size",
        type=int,
        default=DEFAULT_MAX_VALUE_LIST_SIZE,
        help=f"Maximum allowed length for any expanded value list (default {DEFAULT_MAX_VALUE_LIST_SIZE}).",
    )
    p.add_argument(
        "--list-wrap",
        type=int,
        default=DEFAULT_LIST_WRAP,
        help=f"Wrap scalar lists with this many items per line (default {DEFAULT_LIST_WRAP}).",
    )
    args = p.parse_args()

    with open(args.feature_metadata, "r", encoding="utf-8") as f:
        feature_metadata = json.load(f)

    mono = _load_monotonicity(args.monotonicity)

    action_groups = generate_action_groups(feature_metadata, mono, max_value_list_size=args.max_value_list_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = dumps_pretty(action_groups, indent=2, list_wrap=args.list_wrap)
    out_path.write_text(text, encoding="utf-8")

    print(f"Success! Wrote action groups to: {out_path}")


if __name__ == "__main__":
    main()

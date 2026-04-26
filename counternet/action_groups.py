"""
Shared action-group parsing and rule utilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import torch

__all__ = [
    "ActionGroupSpec",
    "DatasetActionGroups",
    "apply_deterministic_rule",
    "feature_special_values",
    "load_dataset_action_groups",
    "resolve_latent_shift_loadings",
]


def feature_special_values(specials_payload: Any, feat: str) -> List[Any]:
    """Return the sentinel values configured for one feature."""
    if isinstance(specials_payload, dict):
        vals = specials_payload.get(feat, [])
    elif isinstance(specials_payload, list):
        vals = specials_payload
    else:
        vals = []
    return list(vals) if isinstance(vals, list) else []


def apply_deterministic_rule(
    kind: str,
    params: Dict[str, Any],
    x_base: torch.Tensor,
    x_derived: torch.Tensor,
    c_base: torch.Tensor,
) -> torch.Tensor:
    """Apply a Type 1 deterministic dependency rule in raw feature space."""
    del params  # Reserved for future rule-specific parameters.
    if kind == "clip_derived_to_base":
        return torch.minimum(x_derived, c_base)
    if kind == "derived_adds_base_delta":
        return x_derived + (c_base - x_base)
    raise NotImplementedError(f"Unknown deterministic rule kind: '{kind}'.")


def _string_tuple(values: Any) -> Tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return tuple()
    return tuple(str(v) for v in values)


def _ordered_unique(values: List[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return tuple(out)


def _float_mapping(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except Exception as exc:
            raise ValueError(f"Expected numeric mapping value for '{key}', got {value!r}.") from exc
    return out


def _tuple_mapping(raw: Any) -> Dict[str, Tuple[Any, ...]]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Tuple[Any, ...]] = {}
    for key, value in raw.items():
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Expected list/tuple domain for '{key}', got {value!r}.")
        out[str(key)] = tuple(value)
    return out


def resolve_latent_shift_loadings(
    features: Tuple[str, ...],
    rule: Dict[str, Any],
    *,
    increase_only: Optional[Tuple[str, ...]] = None,
    decrease_only: Optional[Tuple[str, ...]] = None,
) -> Dict[str, float]:
    params = dict(rule.get("params", {})) if isinstance(rule, dict) else {}
    explicit = _float_mapping(params.get("loadings", {}))
    if explicit:
        missing = [feat for feat in features if feat not in explicit]
        extra = [feat for feat in explicit.keys() if feat not in features]
        if missing:
            raise ValueError(f"Latent-shift group is missing loadings for features: {missing}")
        if extra:
            raise ValueError(f"Latent-shift group defines loadings for unknown features: {extra}")
        return {feat: float(explicit[feat]) for feat in features}

    inc = set(str(v) for v in (increase_only or ()))
    dec = set(str(v) for v in (decrease_only or ()))
    if features and all(feat in inc for feat in features):
        return {feat: 1.0 for feat in features}
    if features and all(feat in dec for feat in features):
        return {feat: -1.0 for feat in features}

    raise ValueError(
        "Latent-shift group requires explicit rule.params.loadings unless all features are "
        "uniformly increase_only or uniformly decrease_only."
    )


@dataclass(frozen=True)
class ActionGroupSpec:
    """Normalized view of one action-group entry from action_groups.json."""

    id: str
    type: int
    mutable: bool
    features: Tuple[str, ...] = field(default_factory=tuple)
    action_kind: str = "values"
    action_feature: Optional[str] = None
    action_values: Tuple[Any, ...] = field(default_factory=tuple)
    action_scales: Dict[str, float] = field(default_factory=dict)
    action_domains: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)
    base_features: Tuple[str, ...] = field(default_factory=tuple)
    derived_features: Tuple[str, ...] = field(default_factory=tuple)
    rule: Dict[str, Any] = field(default_factory=dict)
    special_values: Any = field(default_factory=list)

    @property
    def base_feature(self) -> Optional[str]:
        if self.base_features:
            return self.base_features[0]
        if self.action_feature is not None:
            return self.action_feature
        if self.features:
            return self.features[0]
        return None

    @property
    def group_features(self) -> Tuple[str, ...]:
        return _ordered_unique(
            list(self.base_features) + list(self.derived_features) + list(self.features)
        )

    @property
    def rule_kind(self) -> str:
        return str(self.rule.get("kind", "")) if isinstance(self.rule, dict) else ""

    @property
    def action_size(self) -> int:
        return len(self.action_values)

    def special_values_for(self, feat: str) -> List[Any]:
        return feature_special_values(self.special_values, feat)

    def is_learnable_values_group(self) -> bool:
        return self.mutable and self.action_kind in ("values", "delta_steps") and self.action_size > 0


@dataclass(frozen=True)
class DatasetActionGroups:
    """Action groups and monotonicity constraints for one dataset"""

    dataset_name: str
    path: Path
    groups: Tuple[ActionGroupSpec, ...]
    increase_only: frozenset[str]
    decrease_only: frozenset[str]


def _parse_action_group_spec(group_id: str, raw_group: Dict[str, Any]) -> ActionGroupSpec:
    ad = raw_group.get("action_domain", {}) if isinstance(raw_group.get("action_domain", {}), dict) else {}
    roles = raw_group.get("roles", {}) if isinstance(raw_group.get("roles", {}), dict) else {}
    rule = raw_group.get("rule", {}) if isinstance(raw_group.get("rule", {}), dict) else {}

    action_feature = ad.get("feature", None)
    if action_feature is not None:
        action_feature = str(action_feature)

    features = _string_tuple(raw_group.get("features", []) or [])
    base_features = _string_tuple(roles.get("base", []) or [])
    derived_features = _string_tuple(roles.get("derived", []) or [])

    if not features and action_feature is not None:
        features = (action_feature,)
    if not features and (base_features or derived_features):
        features = _ordered_unique(list(base_features) + list(derived_features))

    action_kind = str(ad.get("kind", "values"))
    if action_kind == "values":
        raw_values = ad.get("values", [])
    elif action_kind == "delta_steps":
        raw_values = ad.get("deltas", [])
    else:
        raw_values = []
    if not isinstance(raw_values, (list, tuple)):
        raw_values = []

    return ActionGroupSpec(
        id=str(group_id),
        type=int(raw_group.get("type", -1)),
        mutable=bool(raw_group.get("mutable", True)),
        features=features,
        action_kind=action_kind,
        action_feature=action_feature,
        action_values=tuple(raw_values),
        action_scales=_float_mapping(ad.get("scales", {})),
        action_domains=_tuple_mapping(ad.get("domains", {})),
        base_features=base_features,
        derived_features=derived_features,
        rule=dict(rule),
        special_values=raw_group.get("special_values", []) or [],
    )


def _load_monotonicity_sets(dataset_groups: Dict[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    constraints = dataset_groups.get("__constraints__", {})
    if not isinstance(constraints, dict):
        constraints = {}

    monotonicity = constraints.get("monotonicity", {})
    if not isinstance(monotonicity, dict):
        monotonicity = {}

    inc_only = frozenset(str(v) for v in (monotonicity.get("increase_only", []) or []))
    dec_only = frozenset(str(v) for v in (monotonicity.get("decrease_only", []) or []))
    return inc_only, dec_only


def load_dataset_action_groups(action_groups_path: Path, dataset_name: str) -> DatasetActionGroups:
    """Load and normalize all action groups for one dataset."""
    action_groups_path = Path(action_groups_path)
    if not action_groups_path.exists():
        raise FileNotFoundError(f"action_groups.json not found at: {action_groups_path}")

    with action_groups_path.open("r", encoding="utf-8") as f:
        all_groups = json.load(f)

    dataset_groups = all_groups.get(dataset_name, {})
    if not isinstance(dataset_groups, dict):
        raise ValueError(f"Invalid action_groups for dataset '{dataset_name}'")

    inc_only, dec_only = _load_monotonicity_sets(dataset_groups)

    groups: List[ActionGroupSpec] = []
    for group_id in sorted(k for k in dataset_groups.keys() if k != "__constraints__"):
        raw_group = dataset_groups[group_id]
        if not isinstance(raw_group, dict):
            continue
        groups.append(_parse_action_group_spec(group_id, raw_group))

    return DatasetActionGroups(
        dataset_name=str(dataset_name),
        path=action_groups_path,
        groups=tuple(groups),
        increase_only=inc_only,
        decrease_only=dec_only,
    )

"""
Shared actionable runtime metadata and inference-time projection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .action_groups import (
    ActionGroupSpec,
    apply_deterministic_rule,
    load_dataset_action_groups,
)

__all__ = [
    "ActionProjectionDerived",
    "ActionProjectionGroup",
    "ActionProjectionSpec",
    "action_feasibility_mask",
    "build_action_projection_spec",
    "load_supported_action_groups",
    "project_to_actionable",
]


@dataclass(frozen=True)
class ActionProjectionDerived:
    feature: str
    kind: str
    cont_index: Optional[int] = None
    slice: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class ActionProjectionGroup:
    id: str
    type: int
    mutable: bool
    action_kind: str
    features: Tuple[str, ...] = field(default_factory=tuple)
    base_feature: Optional[str] = None
    kind: str = ""
    action_size: int = 0
    cont_index: Optional[int] = None
    slice: Optional[Tuple[int, int]] = None
    num_categories: Optional[int] = None
    raw_domain: torch.Tensor = field(default_factory=lambda: torch.empty((0,), dtype=torch.float32))
    scaled_domain: torch.Tensor = field(default_factory=lambda: torch.empty((0,), dtype=torch.float32))
    domain_indices: torch.Tensor = field(default_factory=lambda: torch.empty((0,), dtype=torch.long))
    pos_map: torch.Tensor = field(default_factory=lambda: torch.empty((0,), dtype=torch.long))
    increase_only: bool = False
    decrease_only: bool = False
    special_values: Tuple[Any, ...] = field(default_factory=tuple)
    derived: Tuple[ActionProjectionDerived, ...] = field(default_factory=tuple)
    rule: Dict[str, Any] = field(default_factory=dict)
    raw_v0: Optional[float] = None
    raw_step: Optional[float] = None


@dataclass(frozen=True)
class ActionProjectionSpec:
    dataset_name: str
    action_groups_path: Path
    scaler_min: float
    scaler_max: float
    groups: Tuple[ActionProjectionGroup, ...] = field(default_factory=tuple)
    increase_only: frozenset[str] = field(default_factory=frozenset)
    decrease_only: frozenset[str] = field(default_factory=frozenset)
    continous_cols: Tuple[str, ...] = field(default_factory=tuple)
    discret_cols: Tuple[str, ...] = field(default_factory=tuple)


def _scaler_bounds(scaler: Any) -> Tuple[float, float]:
    mn = float(torch.as_tensor(getattr(scaler, "min_")).item())
    mx = float(torch.as_tensor(getattr(scaler, "max_")).item())
    return mn, mx


def _scale_raw_continuous(raw: torch.Tensor, *, scaler_min: float, scaler_max: float) -> torch.Tensor:
    denom = float(scaler_max - scaler_min)
    if denom == 0.0:
        denom = 1.0
    mn = torch.tensor(float(scaler_min), device=raw.device, dtype=raw.dtype)
    return (raw - mn) / float(denom)


def _unscale_continuous(value: torch.Tensor, *, scaler_min: float, scaler_max: float) -> torch.Tensor:
    denom = float(scaler_max - scaler_min)
    if denom == 0.0:
        denom = 1.0
    mn = torch.tensor(float(scaler_min), device=value.device, dtype=value.dtype)
    return value * float(denom) + mn


def _resolve_category_domain_indices(
    *,
    feat: str,
    values: Sequence[Any],
    categories: Sequence[Any],
) -> List[int]:
    cat_to_index = {c: j for j, c in enumerate(categories)}
    domain_cat_indices: List[int] = []

    for value in values:
        if value in cat_to_index:
            domain_cat_indices.append(int(cat_to_index[value]))
            continue

        found = False
        for category, index in cat_to_index.items():
            try:
                if float(category) == float(value):
                    domain_cat_indices.append(int(index))
                    found = True
                    break
            except Exception:
                continue

        if not found:
            raise ValueError(
                f"Domain value '{value}' for feature '{feat}' not found in fitted categories."
            )

    return domain_cat_indices


def load_supported_action_groups(
    action_groups_path: Path,
    dataset_name: str,
    *,
    learnable_only: bool = False,
) -> Tuple[Tuple[ActionGroupSpec, ...], frozenset[str], frozenset[str]]:
    dataset_action_groups = load_dataset_action_groups(action_groups_path, dataset_name)
    groups: List[ActionGroupSpec] = []

    for spec in dataset_action_groups.groups:
        gtype = int(spec.type)
        if gtype not in (0, 1, 2):
            if spec.mutable:
                raise NotImplementedError(
                    f"Mutable action group '{spec.id}' in dataset '{dataset_name}' uses unsupported type={gtype}. "
                    "Only Type 0/1/2 groups are supported."
                )
            continue

        if gtype == 0:
            if len(spec.features) != 1:
                raise ValueError(
                    f"Type 0 group '{spec.id}' in dataset '{dataset_name}' must contain exactly one feature."
                )
        else:
            if not spec.base_features:
                raise ValueError(
                    f"Type {gtype} group '{spec.id}' in dataset '{dataset_name}' must define roles.base."
                )
            if not spec.derived_features:
                raise ValueError(
                    f"Type {gtype} group '{spec.id}' in dataset '{dataset_name}' must define roles.derived."
                )
            if len(spec.base_features) != 1:
                raise ValueError(
                    f"Type {gtype} group '{spec.id}' in dataset '{dataset_name}' must use a single base feature."
                )

        if learnable_only and not spec.is_learnable_values_group():
            continue

        if spec.mutable and spec.action_kind not in ("values", "noop"):
            raise NotImplementedError(
                f"Mutable action group '{spec.id}' in dataset '{dataset_name}' uses unsupported action_kind='{spec.action_kind}'."
            )

        groups.append(spec)

    return (
        tuple(groups),
        dataset_action_groups.increase_only,
        dataset_action_groups.decrease_only,
    )


def build_action_projection_spec(
    *,
    action_groups_path: Path,
    dataset_name: str,
    continous_cols: Sequence[str],
    discret_cols: Sequence[str],
    scaler: Any,
    cat_normalizer: Any,
    group_specs: Optional[Sequence[ActionGroupSpec]] = None,
    increase_only: Optional[Iterable[str]] = None,
    decrease_only: Optional[Iterable[str]] = None,
) -> ActionProjectionSpec:
    if group_specs is None:
        group_specs, inc_only, dec_only = load_supported_action_groups(
            action_groups_path,
            dataset_name,
            learnable_only=False,
        )
    else:
        inc_only = frozenset(str(v) for v in (increase_only or []))
        dec_only = frozenset(str(v) for v in (decrease_only or []))

    cont_cols = tuple(str(v) for v in continous_cols)
    disc_cols = tuple(str(v) for v in discret_cols)
    cont_pos = {name: idx for idx, name in enumerate(cont_cols)}
    disc_pos = {name: idx for idx, name in enumerate(disc_cols)}

    categories = list(getattr(cat_normalizer, "categories", []) or [])
    cat_slices = list(getattr(cat_normalizer, "cat_slices", []) or [])
    scaler_min, scaler_max = _scaler_bounds(scaler)

    groups: List[ActionProjectionGroup] = []
    seen_features: set[str] = set()

    for spec in group_specs:
        group_features = tuple(str(v) for v in spec.group_features)
        overlap = seen_features.intersection(group_features)
        if overlap:
            raise ValueError(
                f"Action groups must be disjoint, but dataset '{dataset_name}' reuses features {sorted(overlap)}."
            )
        seen_features.update(group_features)

        base_feature = spec.base_feature
        if base_feature is None:
            raise ValueError(f"Action group '{spec.id}' in dataset '{dataset_name}' is missing a base feature.")

        action_kind = str(spec.action_kind)
        mutable = bool(spec.mutable)
        action_size = 0
        raw_domain = torch.empty((0,), dtype=torch.float32)
        scaled_domain = torch.empty((0,), dtype=torch.float32)
        domain_indices = torch.empty((0,), dtype=torch.long)
        pos_map = torch.empty((0,), dtype=torch.long)
        num_categories: Optional[int] = None
        cont_index: Optional[int] = None
        cat_slice: Optional[Tuple[int, int]] = None
        kind = ""
        raw_v0: Optional[float] = None
        raw_step: Optional[float] = None

        if base_feature in cont_pos:
            kind = "continuous"
            cont_index = int(cont_pos[base_feature])

            if action_kind == "values":
                raw_vals = [float(v) for v in spec.action_values]
                raw_domain = torch.tensor(raw_vals, dtype=torch.float32)
                scaled_domain = _scale_raw_continuous(
                    raw_domain,
                    scaler_min=scaler_min,
                    scaler_max=scaler_max,
                )
                action_size = int(raw_domain.numel())
                if action_size > 0:
                    raw_v0 = float(raw_vals[0])
                    raw_step = float(raw_vals[1] - raw_vals[0]) if action_size > 1 else 1.0

        elif base_feature in disc_pos:
            kind = "categorical"
            disc_index = int(disc_pos[base_feature])
            if disc_index >= len(categories) or disc_index >= len(cat_slices):
                raise ValueError(f"Discrete feature '{base_feature}' is missing categorical metadata.")

            cats = list(categories[disc_index])
            num_categories = int(len(cats))
            cat_slice = tuple(int(v) for v in cat_slices[disc_index])

            if action_kind == "values":
                domain_cat_indices = _resolve_category_domain_indices(
                    feat=base_feature,
                    values=list(spec.action_values),
                    categories=cats,
                )
                domain_indices = torch.tensor(domain_cat_indices, dtype=torch.long)
                scaled_domain = torch.nn.functional.one_hot(
                    domain_indices,
                    num_classes=num_categories,
                ).to(torch.float32)
                pos_map = torch.full((num_categories,), -1, dtype=torch.long)
                for position, cat_index in enumerate(domain_cat_indices):
                    pos_map[int(cat_index)] = int(position)
                action_size = int(domain_indices.numel())
        else:
            raise ValueError(
                f"Feature '{base_feature}' from action group '{spec.id}' is not present in continous_cols/discret_cols."
            )

        derived_items: List[ActionProjectionDerived] = []
        if spec.derived_features:
            if kind != "continuous":
                raise NotImplementedError(
                    f"Group '{spec.id}' in dataset '{dataset_name}' uses derived features with non-continuous base '{base_feature}'."
                )

            for feat in spec.derived_features:
                if feat in cont_pos:
                    derived_items.append(
                        ActionProjectionDerived(
                            feature=str(feat),
                            kind="continuous",
                            cont_index=int(cont_pos[feat]),
                        )
                    )
                    continue

                if feat in disc_pos:
                    raise NotImplementedError(
                        f"Group '{spec.id}' in dataset '{dataset_name}' uses unsupported categorical derived feature '{feat}'."
                    )

                raise ValueError(
                    f"Derived feature '{feat}' from action group '{spec.id}' is not present in continous_cols/discret_cols."
                )

        groups.append(
            ActionProjectionGroup(
                id=str(spec.id),
                type=int(spec.type),
                mutable=mutable,
                action_kind=action_kind,
                features=tuple(str(v) for v in spec.features),
                base_feature=str(base_feature),
                kind=kind,
                action_size=action_size,
                cont_index=cont_index,
                slice=cat_slice,
                num_categories=num_categories,
                raw_domain=raw_domain,
                scaled_domain=scaled_domain,
                domain_indices=domain_indices,
                pos_map=pos_map,
                increase_only=str(base_feature) in inc_only,
                decrease_only=str(base_feature) in dec_only,
                special_values=tuple(spec.special_values_for(base_feature)),
                derived=tuple(derived_items),
                rule=dict(spec.rule),
                raw_v0=raw_v0,
                raw_step=raw_step,
            )
        )

    return ActionProjectionSpec(
        dataset_name=str(dataset_name),
        action_groups_path=Path(action_groups_path),
        scaler_min=float(scaler_min),
        scaler_max=float(scaler_max),
        groups=tuple(groups),
        increase_only=frozenset(str(v) for v in inc_only),
        decrease_only=frozenset(str(v) for v in dec_only),
        continous_cols=cont_cols,
        discret_cols=disc_cols,
    )


def action_feasibility_mask(
    group: ActionProjectionGroup,
    x: torch.Tensor,
    *,
    eps: float = 1e-6,
    ensure_any: bool = False,
) -> torch.Tensor:
    action_size = int(group.action_size)
    device = x.device
    if action_size <= 0:
        return torch.zeros((x.shape[0], 0), dtype=torch.bool, device=device)

    if not (group.increase_only or group.decrease_only):
        return torch.ones((x.shape[0], action_size), dtype=torch.bool, device=device)

    if group.kind == "continuous":
        if group.cont_index is None:
            raise ValueError(f"Continuous group '{group.id}' is missing cont_index.")

        dom = group.scaled_domain.to(device=device, dtype=x.dtype)
        cur = x[:, int(group.cont_index)]

        dom_min = dom.min()
        dom_max = dom.max()
        in_range = (cur >= (dom_min - eps)) & (cur <= (dom_max + eps))

        feasible = torch.ones((x.shape[0], action_size), dtype=torch.bool, device=device)
        if group.increase_only:
            feasible[in_range] = dom.unsqueeze(0) >= (cur[in_range].unsqueeze(1) - eps)
        if group.decrease_only:
            feasible[in_range] = dom.unsqueeze(0) <= (cur[in_range].unsqueeze(1) + eps)
    else:
        if group.slice is None:
            raise ValueError(f"Categorical group '{group.id}' is missing slice.")

        start, end = group.slice
        block = x[:, start:end]
        cur_cat = block.argmax(dim=1)

        pos_map = group.pos_map.to(device=device)
        cur_pos = pos_map[cur_cat]

        pos = torch.arange(action_size, device=device).unsqueeze(0).expand(x.shape[0], -1)
        feasible = torch.ones((x.shape[0], action_size), dtype=torch.bool, device=device)

        known = cur_pos >= 0
        if known.any():
            if group.increase_only:
                feasible[known] = pos[known] >= cur_pos[known].unsqueeze(1)
            if group.decrease_only:
                feasible[known] = pos[known] <= cur_pos[known].unsqueeze(1)

    if ensure_any and feasible.numel() > 0:
        none_ok = ~feasible.any(dim=1)
        if none_ok.any():
            feasible[none_ok] = True

    return feasible


def _project_type0_continuous(
    *,
    x: torch.Tensor,
    cf_soft: torch.Tensor,
    group: ActionProjectionGroup,
    eps: float,
) -> torch.Tensor:
    if group.cont_index is None:
        raise ValueError(f"Continuous group '{group.id}' is missing cont_index.")

    j = int(group.cont_index)
    ref = cf_soft[:, j : j + 1]
    noop = x[:, j : j + 1]
    if group.action_size <= 0 or group.action_kind != "values" or not group.mutable:
        return noop

    dom = group.scaled_domain.to(device=x.device, dtype=x.dtype).unsqueeze(0)
    feasible = action_feasibility_mask(group, x, eps=eps, ensure_any=False)
    inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)

    noop_score = (ref - noop).pow(2).sum(dim=1, keepdim=True)
    action_score = (ref - dom).pow(2)
    action_score = action_score.masked_fill(~feasible, inf)

    scores = torch.cat((noop_score, action_score), dim=1)
    idx = scores.argmin(dim=1)

    selected = noop.squeeze(1).clone()
    mask = idx > 0
    if mask.any():
        action_idx = idx[mask] - 1
        selected[mask] = dom[0, action_idx]
    return selected.unsqueeze(1)


def _project_type0_categorical(
    *,
    x: torch.Tensor,
    cf_soft: torch.Tensor,
    group: ActionProjectionGroup,
    eps: float,
) -> torch.Tensor:
    if group.slice is None:
        raise ValueError(f"Categorical group '{group.id}' is missing slice.")

    start, end = group.slice
    ref = cf_soft[:, start:end]
    noop = x[:, start:end]
    if group.action_size <= 0 or group.action_kind != "values" or not group.mutable:
        return noop

    dom = group.scaled_domain.to(device=x.device, dtype=x.dtype)
    feasible = action_feasibility_mask(group, x, eps=eps, ensure_any=False)
    inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)

    noop_score = (ref - noop).pow(2).sum(dim=1, keepdim=True)
    action_score = (ref.unsqueeze(1) - dom.unsqueeze(0)).pow(2).sum(dim=2)
    action_score = action_score.masked_fill(~feasible, inf)

    scores = torch.cat((noop_score, action_score), dim=1)
    idx = scores.argmin(dim=1)

    selected = noop.clone()
    mask = idx > 0
    if mask.any():
        action_idx = idx[mask] - 1
        selected[mask] = dom[action_idx]
    return selected


def _project_type12_continuous(
    *,
    spec: ActionProjectionSpec,
    x: torch.Tensor,
    cf_soft: torch.Tensor,
    group: ActionProjectionGroup,
    eps: float,
) -> Dict[int, torch.Tensor]:
    if group.cont_index is None:
        raise ValueError(f"Derived-action group '{group.id}' is missing cont_index.")
    if group.action_size <= 0 or group.action_kind != "values" or not group.mutable:
        out = {int(group.cont_index): x[:, int(group.cont_index) : int(group.cont_index) + 1]}
        for derived in group.derived:
            if derived.cont_index is None:
                raise ValueError(f"Derived feature '{derived.feature}' is missing cont_index.")
            j = int(derived.cont_index)
            out[j] = x[:, j : j + 1]
        return out

    kind = str(group.rule.get("kind", ""))
    params = dict(group.rule.get("params", {}))
    if kind == "":
        raise ValueError(f"Group '{group.id}' is missing a deterministic rule.")

    j_base = int(group.cont_index)
    base_ref = cf_soft[:, j_base : j_base + 1]
    base_noop = x[:, j_base : j_base + 1]

    dom_scaled = group.scaled_domain.to(device=x.device, dtype=x.dtype)
    dom_raw = group.raw_domain.to(device=x.device, dtype=x.dtype)
    feasible = action_feasibility_mask(group, x, eps=eps, ensure_any=False)
    inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)

    x_base_raw = _unscale_continuous(
        x[:, j_base : j_base + 1],
        scaler_min=spec.scaler_min,
        scaler_max=spec.scaler_max,
    )

    noop_score = (base_ref - base_noop).pow(2).sum(dim=1, keepdim=True)
    action_score = (base_ref - dom_scaled.unsqueeze(0)).pow(2)

    derived_candidates: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    for derived in group.derived:
        if derived.kind != "continuous" or derived.cont_index is None:
            raise NotImplementedError(
                f"Group '{group.id}' uses unsupported derived feature kind='{derived.kind}'."
            )

        j = int(derived.cont_index)
        ref = cf_soft[:, j : j + 1]
        noop = x[:, j : j + 1]
        noop_score = noop_score + (ref - noop).pow(2).sum(dim=1, keepdim=True)

        x_derived_raw = _unscale_continuous(
            x[:, j : j + 1],
            scaler_min=spec.scaler_min,
            scaler_max=spec.scaler_max,
        )
        raw_candidate = apply_deterministic_rule(
            kind,
            params,
            x_base_raw,
            x_derived_raw,
            dom_raw.unsqueeze(0),
        )
        scaled_candidate = _scale_raw_continuous(
            raw_candidate,
            scaler_min=spec.scaler_min,
            scaler_max=spec.scaler_max,
        )
        action_score = action_score + (ref - scaled_candidate).pow(2)
        derived_candidates.append((j, noop, scaled_candidate))

    action_score = action_score.masked_fill(~feasible, inf)
    scores = torch.cat((noop_score, action_score), dim=1)
    idx = scores.argmin(dim=1)

    chosen: Dict[int, torch.Tensor] = {}
    base_selected = base_noop.squeeze(1).clone()
    mask = idx > 0
    if mask.any():
        action_idx = idx[mask] - 1
        base_selected[mask] = dom_scaled[action_idx]
    chosen[j_base] = base_selected.unsqueeze(1)

    for j, noop, scaled_candidate in derived_candidates:
        selected = noop.squeeze(1).clone()
        if mask.any():
            action_idx = idx[mask] - 1
            selected[mask] = scaled_candidate[mask, action_idx]
        chosen[j] = selected.unsqueeze(1)

    return chosen


def project_to_actionable(
    x: torch.Tensor,
    cf_soft: torch.Tensor,
    spec: ActionProjectionSpec,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if x.shape != cf_soft.shape:
        raise ValueError(f"Expected x and cf_soft to have the same shape, got {x.shape} and {cf_soft.shape}.")

    projected = x.clone()
    for group in spec.groups:
        if group.type in (1, 2) and group.derived:
            updates = _project_type12_continuous(
                spec=spec,
                x=x,
                cf_soft=cf_soft,
                group=group,
                eps=eps,
            )
            for index, value in updates.items():
                projected[:, index : index + 1] = value
            continue

        if group.kind == "continuous":
            if group.cont_index is None:
                raise ValueError(f"Continuous group '{group.id}' is missing cont_index.")
            projected[:, group.cont_index : group.cont_index + 1] = _project_type0_continuous(
                x=x,
                cf_soft=cf_soft,
                group=group,
                eps=eps,
            )
            continue

        if group.kind == "categorical":
            if group.slice is None:
                raise ValueError(f"Categorical group '{group.id}' is missing slice.")
            start, end = group.slice
            projected[:, start:end] = _project_type0_categorical(
                x=x,
                cf_soft=cf_soft,
                group=group,
                eps=eps,
            )
            continue

        raise NotImplementedError(f"Unsupported group kind '{group.kind}' for group '{group.id}'.")

    return projected

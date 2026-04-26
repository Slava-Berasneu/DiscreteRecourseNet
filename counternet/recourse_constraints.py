"""Actionability utilities (invalid-action masking + actionability metrics).

The constraints are from `action_groups.json`.

Supported:
  - Type 0 groups
  - Type 1/2 groups with deterministic base-derived rules
  - Type 4 groups with shared latent-shift deltas
  - Monotonicity constraints in `__constraints__/monotonicity`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .import_essentials import *  # torch, json, np, etc.
from .action_groups import (
    ActionGroupSpec,
    apply_deterministic_rule,
    load_dataset_action_groups,
    resolve_latent_shift_loadings,
)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None




def _is_missing(v: Any) -> bool:
    """
    Return True for None/NaN-like values. (missing == missing) to avoid counting "NaN -> NaN" as a change
    """
    if v is None:
        return True
    try:
        return bool(v != v)
    except Exception:
        return False


def _safe_eq(a: Any, b: Any) -> bool:
    if _is_missing(a) and _is_missing(b):
        return True
    return a == b


def _safe_index_in_domain(v: Any, domain: Sequence[Any]) -> int:
    for i, d in enumerate(domain):
        if _safe_eq(v, d):
            return i
    return -1


def _continuous_comparable_mask(
    tensors: Sequence[torch.Tensor],
    specials_per_tensor: Sequence[Sequence[Any]],
    *,
    atol: float,
) -> torch.Tensor:
    if not tensors:
        raise ValueError("Expected at least one tensor for comparability check.")

    device = tensors[0].device
    mask = torch.ones_like(tensors[0], dtype=torch.bool, device=device)
    for t, specials in zip(tensors, specials_per_tensor):
        mask = mask & (~torch.isnan(t))
        sp = [float(s) for s in specials if _safe_float(s) is not None]
        if sp:
            sp_t = torch.tensor(sp, device=device, dtype=t.dtype).view(1, -1)
            mask = mask & (
                ~torch.isclose(t.view(-1, 1), sp_t, atol=atol, rtol=0.0).any(dim=1)
            )
    return mask


def _decode_cat_labels(
    feature: str,
    X: torch.Tensor,
    *,
    disc_idx: Dict[str, int],
    cat_slices: Sequence[Tuple[int, int]],
    categories: Sequence[Sequence[Any]],
) -> List[Any]:
    i = disc_idx[feature]
    (s, e) = cat_slices[i]
    block = X[:, s:e]
    idx = block.argmax(dim=1).detach().cpu().numpy().tolist()
    cats = categories[i]
    out: List[Any] = []
    for j in idx:
        try:
            out.append(cats[int(j)])
        except Exception:
            out.append(None)
    return out


def _continuous_domain_membership(
    values: torch.Tensor,
    domain: Sequence[Any],
    *,
    atol: float,
) -> torch.Tensor:
    dom = [float(v) for v in domain if _safe_float(v) is not None]
    if not dom:
        return torch.zeros_like(values, dtype=torch.bool)
    dom_t = torch.tensor(dom, device=values.device, dtype=values.dtype).view(1, -1)
    return torch.isclose(values.view(-1, 1), dom_t, atol=atol, rtol=0.0).any(dim=1)


def _evaluate_feature_change(
    *,
    feat: str,
    mutable: bool,
    action_kind: str,
    action_domain_values: Optional[Sequence[Any]],
    specials: Sequence[Any],
    x: torch.Tensor,
    cf: torch.Tensor,
    x_cont_raw: torch.Tensor,
    cf_cont_raw: torch.Tensor,
    cont_idx: Dict[str, int],
    disc_idx: Dict[str, int],
    cat_slices: Sequence[Tuple[int, int]],
    categories: Sequence[Sequence[Any]],
    inc_only: Sequence[str],
    dec_only: Sequence[str],
    atol: float,
) -> Optional[Dict[str, torch.Tensor]]:
    n = x.size(0)
    device = x.device

    if feat in cont_idx:
        j = cont_idx[feat]
        xj = x_cont_raw[:, j]
        cj = cf_cont_raw[:, j]

        x_nan = torch.isnan(xj)
        c_nan = torch.isnan(cj)
        changed = ((~x_nan) & (~c_nan) & ((cj - xj).abs() > atol)) | (x_nan ^ c_nan)

        comparable = _continuous_comparable_mask([xj, cj], [specials, specials], atol=atol)
        mono_viol = torch.zeros((n,), dtype=torch.bool, device=device)
        immut_viol = torch.zeros((n,), dtype=torch.bool, device=device)

        if (not mutable) or (action_kind == 'noop'):
            immut_viol = changed
        else:
            mono_check_mask = changed & comparable
            if (feat in inc_only or feat in dec_only) and mono_check_mask.any():
                if feat in inc_only:
                    mono_viol = mono_check_mask & (cj + atol < xj)
                else:
                    mono_viol = mono_check_mask & (cj - atol > xj)

        pre_valid = changed & comparable & (~mono_viol) & (~immut_viol)
        return {
            'changed': changed,
            'mono_viol': mono_viol,
            'immut_viol': immut_viol,
            'pre_valid': pre_valid,
        }

    if feat in disc_idx:
        x_lab = _decode_cat_labels(
            feat, x, disc_idx=disc_idx, cat_slices=cat_slices, categories=categories
        )
        c_lab = _decode_cat_labels(
            feat, cf, disc_idx=disc_idx, cat_slices=cat_slices, categories=categories
        )

        changed = torch.tensor(
            [not _safe_eq(xl, cl) for xl, cl in zip(x_lab, c_lab)],
            device=device,
            dtype=torch.bool,
        )

        comparable = torch.tensor(
            [
                (not _is_missing(xl))
                and (not _is_missing(cl))
                and (not any(_safe_eq(xl, s) for s in specials))
                and (not any(_safe_eq(cl, s) for s in specials))
                for xl, cl in zip(x_lab, c_lab)
            ],
            device=device,
            dtype=torch.bool,
        )

        mono_viol = torch.zeros((n,), dtype=torch.bool, device=device)
        immut_viol = torch.zeros((n,), dtype=torch.bool, device=device)

        if (not mutable) or (action_kind == 'noop'):
            immut_viol = changed
        else:
            mono_check_mask = changed & comparable
            if (feat in inc_only or feat in dec_only) and mono_check_mask.any():
                allowed = list(action_domain_values or categories[disc_idx[feat]])
                x_pos = torch.tensor([_safe_index_in_domain(v, allowed) for v in x_lab], device=device)
                c_pos = torch.tensor([_safe_index_in_domain(v, allowed) for v in c_lab], device=device)

                unmapped = (x_pos < 0) | (c_pos < 0)
                if feat in inc_only:
                    mono_viol = mono_check_mask & (unmapped | (c_pos < x_pos))
                else:
                    mono_viol = mono_check_mask & (unmapped | (c_pos > x_pos))

        pre_valid = changed & comparable & (~mono_viol) & (~immut_viol)
        return {
            'changed': changed,
            'mono_viol': mono_viol,
            'immut_viol': immut_viol,
            'pre_valid': pre_valid,
        }

    return None


def _evaluate_group_rule_violation(
    *,
    group: ActionGroupSpec,
    inc_only: Sequence[str],
    dec_only: Sequence[str],
    x_cont_raw: torch.Tensor,
    cf_cont_raw: torch.Tensor,
    cont_idx: Dict[str, int],
    atol: float,
    device: torch.device,
) -> Tuple[torch.Tensor, bool]:
    n = x_cont_raw.size(0)
    zeros = torch.zeros((n,), dtype=torch.bool, device=device)
    gtype = int(group.type)

    if gtype == 4:
        features = list(group.group_features)
        if not features or group.action_kind != "delta_steps":
            return zeros, False
        if any(feat not in cont_idx for feat in features):
            return zeros, False

        try:
            loadings_map = resolve_latent_shift_loadings(
                tuple(features),
                group.rule,
                increase_only=tuple(str(v) for v in inc_only),
                decrease_only=tuple(str(v) for v in dec_only),
            )
        except Exception:
            return zeros, False

        deltas = [float(v) for v in group.action_values]
        if not deltas:
            return zeros, False

        delta_t = torch.tensor(deltas, device=device, dtype=x_cont_raw.dtype).view(1, -1)
        any_changed = zeros.clone()
        outside_domain_changed = zeros.clone()
        possible = torch.ones((n, len(deltas)), dtype=torch.bool, device=device)
        rule_atol = max(float(atol), 1e-5)

        for feat in features:
            if feat not in group.action_scales or feat not in group.action_domains:
                return zeros, False

            xj = x_cont_raw[:, cont_idx[feat]]
            cj = cf_cont_raw[:, cont_idx[feat]]
            changed = ~torch.isclose(cj, xj, atol=rule_atol, rtol=1e-6)
            any_changed |= changed

            dom = list(group.action_domains.get(feat, ()))
            x_valid = _continuous_domain_membership(xj, dom, atol=rule_atol)
            c_valid = _continuous_domain_membership(cj, dom, atol=rule_atol)
            outside_domain_changed |= changed & (~x_valid | ~c_valid)

            step = float(group.action_scales[feat]) * float(loadings_map[feat])
            candidate = xj.view(-1, 1) + delta_t * float(step)
            possible &= torch.isclose(cj.view(-1, 1), candidate, atol=rule_atol, rtol=1e-6)

        violations = any_changed & (outside_domain_changed | (~possible.any(dim=1)))
        return violations, True

    base_feats = list(group.base_features)
    derived_feats = list(group.derived_features)
    rule = dict(group.rule)
    kind = group.rule_kind

    if len(base_feats) != 1 or len(derived_feats) == 0 or kind == '':
        return zeros, False

    base_feat = str(base_feats[0])
    if base_feat not in cont_idx:
        return zeros, False

    x_base = x_cont_raw[:, cont_idx[base_feat]]
    cf_base = cf_cont_raw[:, cont_idx[base_feat]]
    base_specials = group.special_values_for(base_feat)

    any_rule_viol = zeros.clone()
    supported = False
    # Small comparison tolerance to account for floating-point errors
    rule_atol = max(float(atol), 1e-5)

    for d_feat in derived_feats:
        d_feat = str(d_feat)
        if d_feat not in cont_idx:
            continue

        supported = True
        x_derived = x_cont_raw[:, cont_idx[d_feat]]
        cf_derived = cf_cont_raw[:, cont_idx[d_feat]]
        derived_specials = group.special_values_for(d_feat)

        if kind == 'clip_derived_to_base':
            comparable = _continuous_comparable_mask(
                [x_derived, cf_base, cf_derived],
                [derived_specials, base_specials, derived_specials],
                atol=atol,
            )
        elif kind == 'derived_adds_base_delta':
            comparable = _continuous_comparable_mask(
                [x_base, x_derived, cf_base, cf_derived],
                [base_specials, derived_specials, base_specials, derived_specials],
                atol=atol,
            )
        else:
            supported = False
            continue

        if kind == 'derived_adds_base_delta':
            # Evaluate part-whole rules as a one-way consistency check:
            # if the base feature changes, the derived feature must increase
            # by at least that amount. Derived changes without a base change
            # are allowed, since other latent causes may also affect the total.
            base_delta = cf_base - x_base
            derived_delta = cf_derived - x_derived
            base_changed = comparable & (~torch.isclose(
                cf_base,
                x_base,
                atol=rule_atol,
                rtol=1e-6,
            ))
            any_rule_viol |= base_changed & (derived_delta < (base_delta - rule_atol))
            continue

        expected = apply_deterministic_rule(kind, rule, x_base, x_derived, cf_base)

        any_rule_viol |= comparable & (~torch.isclose(
            cf_derived,
            expected,
            atol=rule_atol,
            rtol=1e-6,
        ))

    return any_rule_viol, supported

def _resolve_action_groups_path(configs: Dict[str, Any]) -> Optional[Path]:
    candidates: List[Path] = []
    if 'action_groups_path' in configs and configs['action_groups_path'] is not None:
        candidates.append(Path(configs['action_groups_path']))
    candidates.extend([
        Path('assets/actions/action_groups.json'),
    ])
    for p in candidates:
        try:
            if p.is_file():
                return p
        except Exception:
            continue
    return None


@dataclass
class ActionabilityMetrics:
    """Actionability metrics computer over the evaluated subset

    - immutability constraints (immutable/no-op features must not change)
    - monotonicity constraints (increase-only / decrease-only)
    - Type 1/2 deterministic group-rule constraints
    - Type 4 shared-delta consistency constraints
      - `clip_derived_to_base` is checked as an exact consistency rule
      - `derived_adds_base_delta` is checked as a one-way lower bound
      - `latent_shift` is checked by existence of one allowed common delta across all group features

    Fields:
      - actionability_rate: fraction of samples with no violations.
      - monotonicity_violation_rate: fraction of samples with >=1 monotonicity violation.
      - immutability_violation_rate: fraction of samples with >=1 immutable-feature change.
      - group_rule_violation_rate: fraction of samples with >=1 Type 1/2/4 rule violation.
      - avg_num_violations: mean number of violations per sample.
      - avg_num_changes: mean number of changed features per sample.
      - valid_change_rate: among all changes, fraction that are valid
    """
    actionability_rate: float
    monotonicity_violation_rate: float
    immutability_violation_rate: float
    group_rule_violation_rate: float
    avg_num_violations: float
    avg_num_changes: float
    valid_change_rate: float

    group_breakdown: Optional[Dict[str, Dict[str, Any]]] = None

    def as_dict(self) -> Dict[str, float]:
        return {
            'actionability_rate': float(self.actionability_rate),
            'monotonicity_violation_rate': float(self.monotonicity_violation_rate),
            'immutability_violation_rate': float(self.immutability_violation_rate),
            'group_rule_violation_rate': float(self.group_rule_violation_rate),
            'avg_num_actionability_violations': float(self.avg_num_violations),
            'avg_num_actionability_changes': float(self.avg_num_changes),
            'valid_change_rate': float(self.valid_change_rate),
        }

def compute_actionability_metrics(
    pred_model: Any,
    x: torch.Tensor,
    cf: torch.Tensor,
    *,
    action_groups_path: Path,
    dataset_name: str,
    only_mask: Optional[torch.Tensor] = None,
    atol: float = 1e-6,
) -> ActionabilityMetrics:
    """Compute actionability metrics for (x -> cf).

    Supported:
      - Type 0 singleton groups
      - Type 1/2 groups with rules:
        - `clip_derived_to_base`: exact consistency
        - `derived_adds_base_delta`: conditional lower-bound consistency
      - Type 4 groups with a shared allowed latent-shift delta
    """
    if only_mask is not None:
        x = x[only_mask]
        cf = cf[only_mask]
        if x.numel() == 0:
            return ActionabilityMetrics(
                actionability_rate=float('nan'),
                monotonicity_violation_rate=float('nan'),
                immutability_violation_rate=float('nan'),
                group_rule_violation_rate=float('nan'),
                avg_num_violations=float('nan'),
                avg_num_changes=float('nan'),
                valid_change_rate=float('nan'),
                group_breakdown={},
            )

    dataset_action_groups = load_dataset_action_groups(action_groups_path, dataset_name)
    inc_only = set(dataset_action_groups.increase_only)
    dec_only = set(dataset_action_groups.decrease_only)

    # Continuous raw values
    cat_idx = pred_model.cat_normalizer.cat_idx
    x_cont_raw = pred_model.scaler.inverse_transform(x[:, :cat_idx])
    cf_cont_raw = pred_model.scaler.inverse_transform(cf[:, :cat_idx])

    cat_slices = getattr(pred_model.cat_normalizer, 'cat_slices', [])
    categories = getattr(pred_model.cat_normalizer, 'categories', [])
    disc_cols = list(getattr(pred_model, 'discret_cols', []))
    cont_cols = list(getattr(pred_model, 'continous_cols', []))
    cont_idx = {f: i for i, f in enumerate(cont_cols)}
    disc_idx = {f: i for i, f in enumerate(disc_cols)}

    n = x.size(0)
    group_breakdown: Dict[str, Dict[str, Any]] = {}

    any_mono_violation = torch.zeros((n,), dtype=torch.bool, device=x.device)
    any_immut_violation = torch.zeros((n,), dtype=torch.bool, device=x.device)
    any_rule_violation = torch.zeros((n,), dtype=torch.bool, device=x.device)
    num_violations = torch.zeros((n,), dtype=torch.long, device=x.device)
    num_changes = torch.zeros((n,), dtype=torch.long, device=x.device)
    num_valid_changes = torch.zeros((n,), dtype=torch.long, device=x.device)

    for spec in dataset_action_groups.groups:
        gname = spec.id
        gtype = int(spec.type)
        if gtype not in (0, 1, 2, 4):
            continue

        mutable = bool(spec.mutable)
        kind = str(spec.action_kind)
        group_features = list(spec.group_features)

        if gtype == 0:
            if len(group_features) != 1:
                continue
        elif len(group_features) == 0:
            continue

        base_feat = spec.base_feature or group_features[0]

        g_changed = torch.zeros((n,), dtype=torch.bool, device=x.device)
        g_mono_viol = torch.zeros((n,), dtype=torch.bool, device=x.device)
        g_immut_viol = torch.zeros((n,), dtype=torch.bool, device=x.device)
        g_valid_change = torch.zeros((n,), dtype=torch.bool, device=x.device)
        g_rule_viol = torch.zeros((n,), dtype=torch.bool, device=x.device)
        feature_pre_valids: List[torch.Tensor] = []

        for feat in group_features:
            feat_specials = spec.special_values_for(feat)
            if gtype == 4:
                feat_action_kind = 'noop' if ((not mutable) or kind == 'noop') else kind
            else:
                feat_action_kind = kind if feat == base_feat else ('noop' if ((not mutable) or kind == 'noop') else 'values')
            feat_domain_values = list(spec.action_values) if feat == base_feat else None

            stats = _evaluate_feature_change(
                feat=feat,
                mutable=mutable,
                action_kind=feat_action_kind,
                action_domain_values=feat_domain_values,
                specials=feat_specials,
                x=x,
                cf=cf,
                x_cont_raw=x_cont_raw,
                cf_cont_raw=cf_cont_raw,
                cont_idx=cont_idx,
                disc_idx=disc_idx,
                cat_slices=cat_slices,
                categories=categories,
                inc_only=inc_only,
                dec_only=dec_only,
                atol=atol,
            )
            if stats is None:
                continue

            g_changed |= stats['changed']
            g_mono_viol |= stats['mono_viol']
            g_immut_viol |= stats['immut_viol']
            any_mono_violation |= stats['mono_viol']
            any_immut_violation |= stats['immut_viol']
            num_changes += stats['changed'].long()
            num_violations += stats['mono_viol'].long() + stats['immut_viol'].long()
            feature_pre_valids.append(stats['pre_valid'])

        if gtype in (1, 2, 4) and mutable and kind != 'noop':
            g_rule_viol, rule_supported = _evaluate_group_rule_violation(
                group=spec,
                inc_only=inc_only,
                dec_only=dec_only,
                x_cont_raw=x_cont_raw,
                cf_cont_raw=cf_cont_raw,
                cont_idx=cont_idx,
                atol=atol,
                device=x.device,
            )
        else:
            rule_supported = False

        any_rule_violation |= g_rule_viol
        num_violations += g_rule_viol.long()

        for pre_valid in feature_pre_valids:
            final_valid = pre_valid & (~g_rule_viol)
            g_valid_change |= final_valid
            num_valid_changes += final_valid.long()

        # Per-group breakdown
        changed_cnt = int(g_changed.sum().item())
        mono_viol_cnt = int(g_mono_viol.sum().item())
        immut_viol_cnt = int(g_immut_viol.sum().item())
        rule_viol_cnt = int(g_rule_viol.sum().item())
        valid_change_cnt = int(g_valid_change.sum().item())

        change_rate = changed_cnt / float(n) if n > 0 else float('nan')
        mono_viol_rate = mono_viol_cnt / float(n) if n > 0 else float('nan')
        immut_viol_rate = immut_viol_cnt / float(n) if n > 0 else float('nan')
        rule_viol_rate = rule_viol_cnt / float(n) if n > 0 else float('nan')

        valid_change_rate_g = (valid_change_cnt / float(changed_cnt)) if changed_cnt > 0 else 1.0
        mono_viol_given_change = (mono_viol_cnt / float(changed_cnt)) if changed_cnt > 0 else 0.0
        immut_viol_given_change = (immut_viol_cnt / float(changed_cnt)) if changed_cnt > 0 else 0.0
        rule_viol_given_change = (rule_viol_cnt / float(changed_cnt)) if changed_cnt > 0 else 0.0

        group_breakdown[gname] = {
            'feature': base_feat,
            'features': group_features,
            'type': gtype,
            'mutable': bool(mutable),
            'n_samples': int(n),
            'changed_count': changed_cnt,
            'change_rate': float(change_rate),
            'monotonicity_violation_rate': float(mono_viol_rate),
            'immutability_violation_rate': float(immut_viol_rate),
            'rule_kind': spec.rule_kind,
            'rule_check_supported': bool(rule_supported),
            'rule_violation_rate': float(rule_viol_rate),
            'valid_change_rate_given_change': float(valid_change_rate_g),
            'monotonicity_violation_rate_given_change': float(mono_viol_given_change),
            'immutability_violation_rate_given_change': float(immut_viol_given_change),
            'rule_violation_rate_given_change': float(rule_viol_given_change),
        }

    feasible = ~(any_mono_violation | any_immut_violation | any_rule_violation)
    actionability_rate = feasible.float().mean().item()
    mono_rate = any_mono_violation.float().mean().item()
    immut_rate = any_immut_violation.float().mean().item()
    rule_rate = any_rule_violation.float().mean().item()
    avg_viol = num_violations.float().mean().item()
    avg_chg = num_changes.float().mean().item()
    tot_changes = int(num_changes.sum().item())
    valid_change_rate = (num_valid_changes.sum().float() / float(tot_changes)).item() if tot_changes > 0 else 1.0

    return ActionabilityMetrics(
        actionability_rate=actionability_rate,
        monotonicity_violation_rate=mono_rate,
        immutability_violation_rate=immut_rate,
        group_rule_violation_rate=rule_rate,
        avg_num_violations=avg_viol,
        avg_num_changes=avg_chg,
        valid_change_rate=valid_change_rate,
        group_breakdown=group_breakdown,
    )


def add_actionability_to_results(
    *,
    results: Dict[str, Any],
    pred_model: Any,
    configs: Dict[str, Any],
    x: torch.Tensor,
    cf: torch.Tensor,
    y_hat: Optional[torch.Tensor] = None,
) -> None:
    """
    Add actionability metrics to results
    """
    dataset_name = configs.get('dataset_name', None)

    nan_metrics = ActionabilityMetrics(
        actionability_rate=float('nan'),
        monotonicity_violation_rate=float('nan'),
        immutability_violation_rate=float('nan'),
        group_rule_violation_rate=float('nan'),
        avg_num_violations=float('nan'),
        avg_num_changes=float('nan'),
        valid_change_rate=float('nan'),
    )

    if dataset_name is None:
        results.update({k: float('nan') for k in nan_metrics.as_dict().keys()})
        results.setdefault('actionability_group_breakdown', {})
        return

    ag_path = _resolve_action_groups_path(configs)
    if ag_path is None:
        results.update({k: float('nan') for k in nan_metrics.as_dict().keys()})
        results.setdefault('actionability_group_breakdown', {})
        return

    # default: evaluate on predicted negatives
    only_mask = None
    if y_hat is not None:
        only_mask = (y_hat.view(-1) == 0)

    m = compute_actionability_metrics(
        pred_model,
        x,
        cf,
        action_groups_path=ag_path,
        dataset_name=str(dataset_name),
        only_mask=only_mask,
    )
    results.update(m.as_dict())
    results['actionability_group_breakdown'] = m.group_breakdown or {}

"""Actionability utilities (invalid-action masking + actionability metrics).

The constraints are from `action_groups.json`.

Supported:
  - Type 0 groups
  - Monotonicity constraints in `__constraints__/monotonicity`.

Placeholders for Types 1-5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .import_essentials import *  # torch, json, np, etc.


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


def _safe_in_list(v: Any, allowed: Sequence[Any]) -> bool:
    if _is_missing(v):
        return any(_is_missing(a) for a in allowed)
    return v in allowed


def _safe_index_in_domain(v: Any, domain: Sequence[Any]) -> int:
    for i, d in enumerate(domain):
        if _safe_eq(v, d):
            return i
    return -1

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


def load_action_groups_for_dataset(action_groups_path: Path, dataset_name: str) -> Dict[str, Any]:
    with open(action_groups_path, 'r', encoding='utf-8') as f:
        ag = json.load(f)
    if dataset_name not in ag:
        raise KeyError(f"Dataset '{dataset_name}' not found in {action_groups_path}")
    return ag[dataset_name]


def get_monotonicity_sets(dataset_action_groups: Dict[str, Any]) -> Tuple[set, set]:
    cons = dataset_action_groups.get('__constraints__', {})
    mono = cons.get('monotonicity', {})
    inc = set(mono.get('increase_only', []) or [])
    dec = set(mono.get('decrease_only', []) or [])
    return inc, dec


def monotonic_valid_mask_numeric(
    feat: str,
    x_slice: torch.Tensor,
    domain1d: torch.Tensor,
    increase_only: Sequence[str],
    decrease_only: Sequence[str],
    eps: float = 1e-9,
) -> torch.Tensor:
    """Return a boolean mask (batch, A_k) for monotonicity.

    Rules:
      - If feature is in increase_only: allow targets >= current.
      - If feature is in decrease_only: allow targets <= current.
      - Else: all valid.

    """
    batch = x_slice.size(0)
    A = domain1d.numel()
    valid = torch.ones((batch, A), dtype=torch.bool, device=x_slice.device)

    inc_set = set(increase_only)
    dec_set = set(decrease_only)
    if feat not in inc_set and feat not in dec_set:
        return valid

    # If current value is outside the domain range (sentinel like -9), disable monotonicity for that sample
    dom_min = domain1d.min()
    dom_max = domain1d.max()
    in_range = (x_slice >= dom_min - eps) & (x_slice <= dom_max + eps)

    if feat in inc_set:
        constrained = domain1d.view(1, -1) >= x_slice.view(-1, 1) - eps
    else:
        constrained = domain1d.view(1, -1) <= x_slice.view(-1, 1) + eps

    valid = torch.where(in_range.view(-1, 1), constrained, valid)

    # Ensure at least one valid per constrained row
    all_invalid = in_range & ((~valid).all(dim=1))
    if all_invalid.any():
        diffs = (domain1d.view(1, -1) - x_slice.view(-1, 1)).abs()
        closest = diffs.argmin(dim=1)
        valid = valid.clone()
        valid[all_invalid] = False
        valid.scatter_(1, closest.view(-1, 1), True)
    return valid


@dataclass
class ActionabilityMetrics:
    """Actionability metrics computer over the evaluated subset

    - immutability constraints (immutable/no-op features must not change)
    - monotonicity constraints (increase-only / decrease-only)

    Fields:
      - actionability_rate: fraction of samples with no violations.
      - monotonicity_violation_rate: fraction of samples with >=1 monotonicity violation.
      - immutability_violation_rate: fraction of samples with >=1 immutable-feature change.
      - avg_num_violations: mean number of violations per sample.
      - avg_num_changes: mean number of changed features per sample.
      - valid_change_rate: among all changes, fraction that are valid
    """
    actionability_rate: float
    monotonicity_violation_rate: float
    immutability_violation_rate: float
    avg_num_violations: float
    avg_num_changes: float
    valid_change_rate: float

    group_breakdown: Optional[Dict[str, Dict[str, Any]]] = None

    def as_dict(self) -> Dict[str, float]:
        return {
            'actionability_rate': float(self.actionability_rate),
            'monotonicity_violation_rate': float(self.monotonicity_violation_rate),
            'immutability_violation_rate': float(self.immutability_violation_rate),
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
    """Compute Type 0 actionability metrics for (x -> cf).

    Checks whether each changed feature respects immutability + monotonicity constraints.
    """
    if only_mask is not None:
        x = x[only_mask]
        cf = cf[only_mask]
        if x.numel() == 0:
            return ActionabilityMetrics(
                actionability_rate=float('nan'),
                monotonicity_violation_rate=float('nan'),
                immutability_violation_rate=float('nan'),
                avg_num_violations=float('nan'),
                avg_num_changes=float('nan'),
                valid_change_rate=float('nan'),
                group_breakdown={},
            )

    ds_ag = load_action_groups_for_dataset(action_groups_path, dataset_name)
    inc_only, dec_only = get_monotonicity_sets(ds_ag)

    # Continuous raw values
    cat_idx = pred_model.cat_normalizer.cat_idx
    x_cont_raw = pred_model.scaler.inverse_transform(x[:, :cat_idx])
    cf_cont_raw = pred_model.scaler.inverse_transform(cf[:, :cat_idx])

    # Categorical decoding helpers
    cat_slices = getattr(pred_model.cat_normalizer, 'cat_slices', [])
    categories = getattr(pred_model.cat_normalizer, 'categories', [])
    disc_cols = list(getattr(pred_model, 'discret_cols', []))
    cont_cols = list(getattr(pred_model, 'continous_cols', []))
    cont_idx = {f: i for i, f in enumerate(cont_cols)}
    disc_idx = {f: i for i, f in enumerate(disc_cols)}

    def decode_cat(feature: str, X: torch.Tensor) -> List[Any]:
        i = disc_idx[feature]
        (s, e) = cat_slices[i]
        block = X[:, s:e]
        idx = block.argmax(dim=1).detach().cpu().numpy().tolist()
        cats = categories[i]
        out = []
        for j in idx:
            try:
                out.append(cats[int(j)])
            except Exception:
                out.append(None)
        return out

    def _is_special_value(v: Any, specials: Optional[Sequence[Any]]) -> bool:
        if not specials:
            return False
        for s in specials:
            if _safe_eq(v, s):
                return True
        return False

    n = x.size(0)
    group_breakdown: Dict[str, Dict[str, Any]] = {}

    any_mono_violation = torch.zeros((n,), dtype=torch.bool, device=x.device)
    any_immut_violation = torch.zeros((n,), dtype=torch.bool, device=x.device)
    num_violations = torch.zeros((n,), dtype=torch.long, device=x.device)
    num_changes = torch.zeros((n,), dtype=torch.long, device=x.device)
    num_valid_changes = torch.zeros((n,), dtype=torch.long, device=x.device)

    for gname, g in ds_ag.items():
        if gname == '__constraints__':
            continue
        if int(g.get('type', 0)) != 0:
            continue

        feats = g.get('features', [])
        if len(feats) != 1:
            continue
        feat = feats[0]

        mutable = bool(g.get('mutable', True))
        specials = g.get('special_values', None)
        ad = g.get('action_domain', {})
        kind = ad.get('kind', 'values')

        g_changed = torch.zeros((n,), dtype=torch.bool, device=x.device)
        g_mono_viol = torch.zeros((n,), dtype=torch.bool, device=x.device)
        g_immut_viol = torch.zeros((n,), dtype=torch.bool, device=x.device)
        g_valid_change = torch.zeros((n,), dtype=torch.bool, device=x.device)

        # Continuous
        if feat in cont_idx:
            j = cont_idx[feat]
            xj = x_cont_raw[:, j]
            cj = cf_cont_raw[:, j]

            x_nan = torch.isnan(xj)
            c_nan = torch.isnan(cj)
            changed = ((~x_nan) & (~c_nan) & ((cj - xj).abs() > atol)) | (x_nan ^ c_nan)
            g_changed = changed
            num_changes += changed.long()

            # skip NaNs and explicit special values (e.g., -9/-8/-7)
            is_comparable = (~x_nan) & (~c_nan)
            if specials:
                sp = [float(s) for s in specials if _safe_float(s) is not None]
                if sp:
                    sp_t = torch.tensor(sp, device=x.device, dtype=torch.float).view(1, -1)
                    is_comparable = is_comparable & (
                        ~torch.isclose(xj.view(-1, 1), sp_t, atol=atol, rtol=0.0).any(dim=1)
                    ) & (
                        ~torch.isclose(cj.view(-1, 1), sp_t, atol=atol, rtol=0.0).any(dim=1)
                    )

            if (not mutable) or (kind == 'noop'):
                viol = changed
                g_immut_viol = viol
                any_immut_violation |= viol
                num_violations += viol.long()
            else:
                # Check monotonicity for any comparable change
                mono_check_mask = changed & is_comparable
                if (feat in inc_only or feat in dec_only) and mono_check_mask.any():
                    if feat in inc_only:
                        mono_viol = mono_check_mask & (cj + atol < xj)
                    else:
                        mono_viol = mono_check_mask & (cj - atol > xj)
                    g_mono_viol = mono_viol
                    any_mono_violation |= mono_viol
                    num_violations += mono_viol.long()

                valid_change = changed & is_comparable & (~g_mono_viol)
                g_valid_change = valid_change
                num_valid_changes += valid_change.long()

        # Categorical one-hot
        elif feat in disc_idx:
            x_lab = decode_cat(feat, x)
            c_lab = decode_cat(feat, cf)

            changed = torch.tensor(
                [not _safe_eq(xl, cl) for xl, cl in zip(x_lab, c_lab)],
                device=x.device,
                dtype=torch.bool,
            )
            g_changed = changed
            num_changes += changed.long()

            is_comparable = torch.tensor(
                [not _is_missing(xl) and not _is_missing(cl) and
                 (not _is_special_value(xl, specials)) and (not _is_special_value(cl, specials))
                 for xl, cl in zip(x_lab, c_lab)],
                device=x.device,
                dtype=torch.bool,
            )

            if (not mutable) or (kind == 'noop'):
                viol = changed
                g_immut_viol = viol
                any_immut_violation |= viol
                num_violations += viol.long()
            else:
                mono_check_mask = changed & is_comparable
                if (feat in inc_only or feat in dec_only) and mono_check_mask.any():
                    allowed = list(ad.get('values', []))  # ordering = list order
                    x_pos = torch.tensor([_safe_index_in_domain(v, allowed) for v in x_lab], device=x.device)
                    c_pos = torch.tensor([_safe_index_in_domain(v, allowed) for v in c_lab], device=x.device)

                    unmapped = (x_pos < 0) | (c_pos < 0)
                    if feat in inc_only:
                        mono_viol = mono_check_mask & (unmapped | (c_pos < x_pos))
                    else:
                        mono_viol = mono_check_mask & (unmapped | (c_pos > x_pos))

                    g_mono_viol = mono_viol
                    any_mono_violation |= mono_viol
                    num_violations += mono_viol.long()

                valid_change = changed & is_comparable & (~g_mono_viol)
                g_valid_change = valid_change
                num_valid_changes += valid_change.long()

        else:
            continue

        # Per-group breakdown
        changed_cnt = int(g_changed.sum().item())
        mono_viol_cnt = int(g_mono_viol.sum().item())
        immut_viol_cnt = int(g_immut_viol.sum().item())
        valid_change_cnt = int(g_valid_change.sum().item())

        change_rate = changed_cnt / float(n) if n > 0 else float('nan')
        mono_viol_rate = mono_viol_cnt / float(n) if n > 0 else float('nan')
        immut_viol_rate = immut_viol_cnt / float(n) if n > 0 else float('nan')

        valid_change_rate_g = (valid_change_cnt / float(changed_cnt)) if changed_cnt > 0 else 1.0
        mono_viol_given_change = (mono_viol_cnt / float(changed_cnt)) if changed_cnt > 0 else 0.0
        immut_viol_given_change = (immut_viol_cnt / float(changed_cnt)) if changed_cnt > 0 else 0.0

        group_breakdown[gname] = {
            'feature': feat,
            'type': int(g.get('type', 0)),
            'mutable': bool(mutable),
            'n_samples': int(n),
            'changed_count': changed_cnt,
            'change_rate': float(change_rate),
            'monotonicity_violation_rate': float(mono_viol_rate),
            'immutability_violation_rate': float(immut_viol_rate),
            'valid_change_rate_given_change': float(valid_change_rate_g),
            'monotonicity_violation_rate_given_change': float(mono_viol_given_change),
            'immutability_violation_rate_given_change': float(immut_viol_given_change),
        }

    feasible = ~(any_mono_violation | any_immut_violation)
    actionability_rate = feasible.float().mean().item()
    mono_rate = any_mono_violation.float().mean().item()
    immut_rate = any_immut_violation.float().mean().item()
    avg_viol = num_violations.float().mean().item()
    avg_chg = num_changes.float().mean().item()
    tot_changes = int(num_changes.sum().item())
    valid_change_rate = (num_valid_changes.sum().float() / float(tot_changes)).item() if tot_changes > 0 else 1.0

    return ActionabilityMetrics(
        actionability_rate=actionability_rate,
        monotonicity_violation_rate=mono_rate,
        immutability_violation_rate=immut_rate,
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

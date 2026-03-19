__all__ = ['LinearBlock', 'MultilayerPerception', 'BaselinePredictiveModel', 'CounterNetModel', 'CounterNetProjectionModel', 'DiscreteRecourseNetModel']

# Cell
from .import_essentials import *
from .utils import *
from .training_module import PredictiveTrainingModule, CFNetTrainingModule
from .action_groups import (
    ActionGroupSpec,
    apply_deterministic_rule,
    load_dataset_action_groups,
)

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Comes from 02b_counter_net.ipynb, cell
class LinearBlock(pl.LightningModule):
    def __init__(self, input_dim, out_dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)

class MultilayerPerception(pl.LightningModule):
    def __init__(self, dims=[3, 100, 10], dropout=0.3):
        super().__init__()
        layers  = []
        num_blocks = len(dims)
        for i in range(1, num_blocks):
            layers += [
                LinearBlock(dims[i-1], dims[i], dropout=dropout)
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Comes from 02b_counter_net.ipynb, cell
class BaselinePredictiveModel(PredictiveTrainingModule):
    def __init__(self, config):
        super().__init__(config)
        assert self.enc_dims[-1] == self.dec_dims[0], \
            f"(enc_dims[-1]={self.enc_dims[-1]}) != (dec_dims[0]={self.dec_dims[0]})"
        self.model = nn.Sequential(
            MultilayerPerception(self.enc_dims, self.dropout),
            MultilayerPerception(self.dec_dims, self.dropout),
            nn.Linear(self.dec_dims[-1], 1)
        )

    def model_forward(self, x):
        # x = ([],)
        x, = x
        y_hat = torch.sigmoid(self.model(x))
        return torch.squeeze(y_hat, -1)

# Comes from 02b_counter_net.ipynb, cell
class CounterNetModel(CFNetTrainingModule):
    def __init__(self, config):
        super().__init__(config)
        assert self.enc_dims[-1] == self.dec_dims[0]
        assert self.enc_dims[-1] == self.exp_dims[0]

        self.encoder_model = MultilayerPerception(self.enc_dims)
        # predictor
        self.predictor = MultilayerPerception(self.dec_dims)
        self.pred_linear = nn.Linear(self.dec_dims[-1], 1)
        # explainer
        exp_dims = list(self.exp_dims)
        exp_dims[0] = self.exp_dims[0] + self.dec_dims[-1]

        self.explainer = nn.Sequential(
            MultilayerPerception(exp_dims),
            nn.Linear(self.exp_dims[-1], self.enc_dims[0])
        )

    def model_forward(self, x):
        x = self.encoder_model(x)
        # predicted y_hat
        pred = self.predictor(x)
        y_hat = torch.sigmoid(self.pred_linear(pred))
        # counterfactual example
        x = torch.cat((x, pred), -1)
        c = self.explainer(x)
        return torch.squeeze(y_hat, -1), c

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predictor-only forward (no CF generation)."""
        z = self.encoder_model(x)
        pred = self.predictor(z)
        y_hat = torch.sigmoid(self.pred_linear(pred))
        return torch.squeeze(y_hat, -1)

    def configure_optimizers(self):
        # Separate optimizers to avoid explainer loss affecting predictor weights.
        pred_params = list(self.encoder_model.parameters()) + list(self.predictor.parameters()) + list(self.pred_linear.parameters())
        exp_params = list(self.explainer.parameters())
        opt_1 = torch.optim.Adam(pred_params, lr=self.lr)
        opt_2 = torch.optim.Adam(exp_params, lr=self.lr)
        return (opt_1, opt_2)


class CounterNetProjectionModel(CounterNetModel):
    """CounterNet actionability baseline idea using projection of immutable features

    After generating a counterfactual, we project immutable coordinates back to
    the original input before computing losses or returning the CF.
    """

    def __init__(self, config):
        if 'immutable_columns' not in config:
            raise ValueError(
                "CounterNetProjectionModel requires 'immutable_columns' in the model config."
            )
        super().__init__(config)
        self.immutable_columns: List[str] = [str(c) for c in (config.get('immutable_columns') or [])]
        self.register_buffer('_immutable_mask', torch.empty(0, dtype=torch.bool), persistent=False)

    def prepare_data(self):
        super().prepare_data()

        feature_names = set(self.continous_cols) | set(self.discret_cols)
        unknown = sorted(set(self.immutable_columns) - feature_names)
        if unknown:
            raise ValueError(
                f"Unknown immutable_columns for dataset '{getattr(self.hparams, 'dataset_name', '')}': {unknown}"
            )

        mask = torch.zeros((self.enc_dims[0],), dtype=torch.bool)
        cont_pos = {name: idx for idx, name in enumerate(self.continous_cols)}
        disc_pos = {name: idx for idx, name in enumerate(self.discret_cols)}

        for name in self.immutable_columns:
            if name in cont_pos:
                mask[cont_pos[name]] = True
                continue

            disc_idx = disc_pos[name]
            start, end = self.cat_normalizer.cat_slices[disc_idx]
            mask[start:end] = True

        self._immutable_mask = mask

    def _project_to_feasible(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Project immutable coordinates back to the original input
        """
        if self._immutable_mask.numel() == 0 or not self._immutable_mask.any().item():
            return c

        if x.shape != c.shape:
            raise ValueError(f"Expected x and c to have the same shape, got {x.shape} and {c.shape}.")

        if self._immutable_mask.shape[0] != c.shape[1]:
            raise RuntimeError(
                f"Immutable mask width {self._immutable_mask.shape[0]} does not match input width {c.shape[1]}."
            )

        mask = self._immutable_mask.to(device=c.device).view(1, -1)
        return torch.where(mask, x, c)

    def forward(self, x, hard: bool = False):
        y, c = self.model_forward(x)
        c = self.cat_normalizer.normalize(c, hard=hard)
        c = self._project_to_feasible(x, c)
        return y, c

    def generate_cf(self, x, clamp=False):
        self.freeze()
        _, c = self.model_forward(x)
        if clamp:
            c = torch.clamp(c, 0., 1.)
        c = self.cat_normalizer.normalize(c, hard=True)
        return self._project_to_feasible(x, c)


class DiscreteRecourseNetModel(CFNetTrainingModule):
    """DiscreteRecourseNet: a generator of actionable counterfactuals, 
    with gumbel-softmax straight-through sampling on mask and choice networks, 
    and invalid action masking for monotonicity constraints.

    Currently implemented:
      - Type 0 (singleton) action groups
      - Type 1/2 (base + derived) action groups

    Expected action_groups.json schema:
      action_groups[dataset][group_id] = {
        "type": 0|1|2,
        "features": ["feature_name", ...],
        "mutable": true/false,
        "action_domain": {"kind": "values"|"noop", "feature": "...", "values": [...]},
        "roles": {"base": [...], "derived": [...]},   # type 1/2 only
        "rule": {"kind": "...", "params": {...}},      # type 1/2 only
      }
      action_groups[dataset]["__constraints__"]["monotonicity"] = {
        "increase_only": [...], "decrease_only": [...]
      }

    Notes:
      - Domains in action_groups.json are in feature space (raw values for continuous, category labels for discrete).
      - This model maps those domains into the preprocessed input space using the fitted MinMaxScaler and OneHotEncoder
        categories stored in CategoricalNormalizer.
      - Type 1/2 groups act on the base feature only, derived features are updated using _apply_derived_constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assert self.enc_dims[-1] == self.dec_dims[0]

        # encoder + predictor networks
        self.encoder_model = MultilayerPerception(self.enc_dims)
        self.predictor = MultilayerPerception(self.dec_dims)
        self.pred_linear = nn.Linear(self.dec_dims[-1], 1)

        # discrete policy hyperparams
        self.gumbel_tau_mask: float = float(config.get("gumbel_tau_mask"))
        self.gumbel_tau_choice: float = float(config.get("gumbel_tau_choice"))
        self.mask_threshold: float = float(config.get("mask_threshold", 0.5))
        self.invalid_logit_value: float = float(config.get("invalid_logit_value", -1e4))
        self._eps: float = float(config.get("mask_eps", 1e-6))

        # Load action groups from action_groups.json (raw feature-space domains).
        self.dataset_name: str = str(config.get("dataset_name", ""))
        default_action_groups_path = Path(__file__).resolve().parents[1] / "assets" / "actions" / "action_groups.json"
        self.action_groups_path = Path(config.get("action_groups_path", default_action_groups_path))

        self._group_specs: List[ActionGroupSpec] = []
        self._inc_only: set[str] = set()
        self._dec_only: set[str] = set()

        self._parse_action_groups()

        self.num_groups: int = len(self._group_specs)
        if self.num_groups == 0:
            raise ValueError(
                f"No mutable groups found for dataset='{self.dataset_name}' in {self.action_groups_path}. "
                "Run scripts/generate_action_groups.py to produce a full action_groups.json."
            )

        # concatenated logits for the action space over groups
        self._action_sizes: List[int] = [int(spec.action_size) for spec in self._group_specs]
        self._action_offsets: List[Tuple[int, int]] = []
        offset = 0
        for a in self._action_sizes:
            self._action_offsets.append((offset, offset + a))
            offset += a
        self.total_actions: int = offset

        z_dim = self.enc_dims[-1]
        p_dim = self.dec_dims[-1]
        policy_input_dim = z_dim + p_dim # provide both encoder input z and predictor representation pred to the CF generator
        # 2-class per group (apply vs not apply)
        self.mask_head = nn.Linear(policy_input_dim, self.num_groups * 2)
        # logits for each group's action domain, concatenated
        self.choice_head = nn.Linear(policy_input_dim, self.total_actions)

        # Built after prepare_data: scaled domains + feature slices
        self._domains_built: bool = False
        self._sanity_logged: bool = False

        # Soft (expected) counterfactual used for proximity loss during training
        self._last_c_soft: Optional[torch.Tensor] = None

        # Expected (differentiable) action cost for training-time proximity.
        self.action_cost_base: float = float(config.get('action_cost_base'))
        self.action_cost_weights: Dict[str, float] = dict(config.get('action_cost_weights', {}) or {})
        self._last_action_cost: Optional[torch.Tensor] = None


    # Action group parsing/build
    def _parse_action_groups(self) -> None:
        dataset_action_groups = load_dataset_action_groups(
            self.action_groups_path,
            self.dataset_name,
        )
        self._inc_only = set(dataset_action_groups.increase_only)
        self._dec_only = set(dataset_action_groups.decrease_only)

        for spec in dataset_action_groups.groups:
            gtype = int(spec.type)
            if gtype not in (0, 1, 2):
                continue

            if gtype == 0:
                # Type 0
                if len(spec.features) != 1:
                    continue
            elif gtype in (1, 2):
                # Type 1/2: base features define the action domain, derived are updated by constraint
                if not spec.base_features:
                    raise ValueError(
                        f"Type {gtype} group '{spec.id}' in dataset '{self.dataset_name}' "
                        f"must define roles.base"
                    )
                if not spec.derived_features:
                    raise ValueError(
                        f"Type {gtype} group '{spec.id}' in dataset '{self.dataset_name}' "
                        f"must define roles.derived"
                    )
                if len(spec.base_features) != 1:
                    raise ValueError(
                        f"Type {gtype} group '{spec.id}': only single-base groups are currently supported, "
                        f"got base={list(spec.base_features)}"
                    )

            # Ignore groups that are not directly controlled
            if not spec.is_learnable_values_group():
                continue

            self._group_specs.append(spec)

    def _default_cost_weight(self, gid: str, feat: str, kind: str, action_size: int) -> float:
        # can be overridden by config
        if gid in self.action_cost_weights:
            return float(self.action_cost_weights[gid])
        if feat in self.action_cost_weights:
            return float(self.action_cost_weights[feat])

        # normalize continuous domains by number of discrete steps
        if kind == "continuous":
            return 1.0 / max(1, int(action_size) - 1)
        # categorical: normalize by number of categories
        return 1.0 / max(1, int(action_size) - 1)

    def prepare_data(self):
        super().prepare_data()
        self._ensure_action_domains()

    @staticmethod
    def _resolve_category_domain_indices(
        *,
        feat: str,
        values: List[Any],
        categories: List[Any],
    ) -> List[int]:
        """Map categorical action domain values to category indices."""
        cat_to_index = {c: j for j, c in enumerate(categories)}
        domain_cat_indices: List[int] = []

        for value in values:
            if value in cat_to_index:
                domain_cat_indices.append(int(cat_to_index[value]))
                continue

            # Numeric coercion handles JSON int vs numpy scalar mismatches.
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

    def _ensure_action_domains(self) -> None:
        """Build action domain tensors in preprocessed input space.

        Continuous:
          - store (A,) scaled target values using fitted MinMaxScaler
        Categorical:
          - store (A, m) one-hot target blocks using fitted OneHotEncoder categories
          - store (m,) map from category index -> position in domain list (for monotonic constraint masking)
        """
        if self._domains_built:
            return

        cont_cols: List[str] = list(self.continous_cols or [])
        disc_cols: List[str] = list(self.discret_cols or [])
        cont_pos = {f: i for i, f in enumerate(cont_cols)}
        disc_pos = {f: i for i, f in enumerate(disc_cols)}

        categories = getattr(self.cat_normalizer, "categories", [])
        cat_slices = getattr(self.cat_normalizer, "cat_slices", [])

        self._group_meta: List[Dict[str, Any]] = []
        for gi, spec in enumerate(self._group_specs):
            feat = spec.base_feature
            if feat is None:
                raise ValueError("Expected every learnable action group to have a base feature.")
            vals = list(spec.action_values)

            if feat in cont_pos:
                j = int(cont_pos[feat])

                raw = torch.tensor([float(v) for v in vals], dtype=torch.float32)
                # processing.MinMaxScaler with scalar tensor attributes `min_` and `max_` over all continuous values
                denom = (self.scaler.max_ - self.scaler.min_)
                if float(denom) == 0.0:
                    denom = torch.tensor(1.0, dtype=torch.float32)
                dom = (raw - self.scaler.min_) / denom
                self.register_buffer(f"_ag_dom_{gi}", dom)
                
                self.register_buffer(f"_ag_idx_{gi}", torch.arange(int(spec.action_size), dtype=torch.float32))
                w = self._default_cost_weight(spec.id, feat, "continuous", int(spec.action_size))
                raw_vals = [float(v) for v in vals]
                step = float(raw_vals[1] - raw_vals[0]) if len(raw_vals) > 1 else 1.0
                self._group_meta.append(
                    {
                        "id": spec.id,
                        "feature": feat,
                        "kind": "continuous",
                        "cont_index": j,
                        "action_size": int(spec.action_size),
                        "increase_only": feat in self._inc_only,
                        "decrease_only": feat in self._dec_only,
                        "cost_weight": w,
                        "raw_v0": float(raw_vals[0]),
                        "raw_step": step,
                        "special_values": [float(v) for v in spec.special_values_for(feat)],
                    }
                )

            elif feat in disc_pos:
                di = int(disc_pos[feat])
                if di >= len(categories) or di >= len(cat_slices):
                    raise ValueError(f"Discrete feature '{feat}' missing category metadata.")

                cats = list(categories[di])
                domain_cat_indices = self._resolve_category_domain_indices(
                    feat=feat,
                    values=vals,
                    categories=cats,
                )

                domain_idx = torch.tensor(domain_cat_indices, dtype=torch.long)
                m = len(cats)
                dom_onehot = torch.nn.functional.one_hot(domain_idx, num_classes=m).to(torch.float32)
                self.register_buffer(f"_ag_dom_{gi}", dom_onehot)

                pos_map = torch.full((m,), -1, dtype=torch.long)
                for p, ci in enumerate(domain_cat_indices):
                    pos_map[int(ci)] = int(p)
                self.register_buffer(f"_ag_posmap_{gi}", pos_map)

                (s, e) = cat_slices[di]
                self.register_buffer(f"_ag_idx_{gi}", torch.arange(int(spec.action_size), dtype=torch.float32))
                w = self._default_cost_weight(spec.id, feat, "categorical", int(spec.action_size))
                self._group_meta.append(
                    {
                        "id": spec.id,
                        "feature": feat,
                        "kind": "categorical",
                        "slice": (int(s), int(e)),
                        "action_size": int(spec.action_size),
                        "num_categories": int(m),
                        "increase_only": feat in self._inc_only,
                        "decrease_only": feat in self._dec_only,
                        "cost_weight": w,
                        "special_values": [float(v) for v in spec.special_values_for(feat)],
                    }
                )
            else:
                raise ValueError(
                    f"Feature '{feat}' (group '{spec.id}') not found in continous_cols or discret_cols."
                )

            # Resolve derived features for Type 1/2
            derived_feats = list(spec.derived_features)
            rule = dict(spec.rule)
            if derived_feats:
                derived_indices: List[Dict[str, Any]] = []
                for df in derived_feats:
                    if df in cont_pos:
                        derived_indices.append({
                            "feature": df,
                            "kind": "continuous",
                            "cont_index": int(cont_pos[df]),
                        })
                    elif df in disc_pos:
                        di = int(disc_pos[df])
                        if di >= len(cat_slices):
                            raise ValueError(
                                f"Derived feature '{df}' missing category metadata."
                            )
                        (s, e) = cat_slices[di]
                        derived_indices.append({
                            "feature": df,
                            "kind": "categorical",
                            "slice": (int(s), int(e)),
                        })
                    else:
                        raise ValueError(
                            f"Derived feature '{df}' (group '{spec.id}') "
                            f"not found in continous_cols or discret_cols."
                        )
                self._group_meta[-1]["derived"] = derived_indices
                self._group_meta[-1]["rule"] = rule

        # group composition printout (continuous vs categorical)
        if not self._sanity_logged:
            n_cont = sum(1 for mm in self._group_meta if mm.get("kind") == "continuous")
            n_cat = sum(1 for mm in self._group_meta if mm.get("kind") == "categorical")
            n_t0 = sum(1 for mm in self._group_meta if not mm.get("derived"))
            n_t12 = sum(1 for mm in self._group_meta if mm.get("derived"))
            print(
                f"[DiscreteRecourseNet] Loaded {len(self._group_meta)} groups "
                f"(type0={n_t0}, type1+2={n_t12}, "
                f"continuous={n_cont}, categorical={n_cat}) "
                f"for dataset='{self.dataset_name}'."
            )
            if n_cat == 0 and len(disc_cols) > 0:
                print(
                    "[DiscreteRecourseNet][WARN] No categorical groups were loaded even though "
                    "the dataset has discrete columns. Check action_groups.json domains / feature names."
                )
            self._sanity_logged = True

        self._domains_built = True

    def forward(self, x, hard: bool = False):
        """
        CFNetTrainingModule.forward() applies CategoricalNormalizer.normalize() which treats the
        categorical part of `c` as logits and applies softmax. DiscreteRecourseNet constructs
        one-hot categorical blocks for Type-0 categorical actions, so we use `hard=True`
        during training to avoid washing out categorical interventions.
        """
        if self.training:
            hard = True
        return super().forward(x, hard=hard)


    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predictor-only forward."""
        z = self.encoder_model(x)
        pred = self.predictor(z)
        y_hat = torch.sigmoid(self.pred_linear(pred))
        return torch.squeeze(y_hat, -1)

    def configure_optimizers(self):
        # Separate optimizers: predictor vs discrete policy (mask+choice)
        pred_params = list(self.encoder_model.parameters()) + list(self.predictor.parameters()) + list(self.pred_linear.parameters())
        policy_params = list(self.mask_head.parameters()) + list(self.choice_head.parameters())
        opt_1 = torch.optim.Adam(pred_params, lr=self.lr)
        opt_2 = torch.optim.Adam(policy_params, lr=self.lr)
        return (opt_1, opt_2)

    # Invalid action masking for monotonicity constraints
    def _mask_invalid_actions(self, gi: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Mask infeasible actions for one group before sampling/argmax."""
        meta = self._group_meta[gi]
        inc_only = bool(meta.get("increase_only", False))
        dec_only = bool(meta.get("decrease_only", False))
        if not (inc_only or dec_only):
            return logits

        A = logits.shape[-1]
        device = logits.device

        if meta["kind"] == "continuous":
            dom = getattr(self, f"_ag_dom_{gi}")  # (A,)
            cur = x[:, int(meta["cont_index"])]  # (B,)

            dom_min = dom.min()
            dom_max = dom.max()
            in_range = (cur >= (dom_min - self._eps)) & (cur <= (dom_max + self._eps))

            feasible = torch.ones((x.shape[0], A), dtype=torch.bool, device=device)
            if inc_only:
                feasible[in_range] = dom.unsqueeze(0) >= (cur[in_range].unsqueeze(1) - self._eps)
            if dec_only:
                feasible[in_range] = dom.unsqueeze(0) <= (cur[in_range].unsqueeze(1) + self._eps)

        else:
            (s, e) = meta["slice"]
            block = x[:, s:e]
            cur_cat = block.argmax(dim=1)

            pos_map = getattr(self, f"_ag_posmap_{gi}")  # (m,)
            cur_pos = pos_map[cur_cat]  # (B,)

            pos = torch.arange(A, device=device).unsqueeze(0).expand(x.shape[0], -1)
            feasible = torch.ones((x.shape[0], A), dtype=torch.bool, device=device)

            known = cur_pos >= 0
            if known.any():
                if inc_only:
                    feasible[known] = pos[known] >= cur_pos[known].unsqueeze(1)
                if dec_only:
                    feasible[known] = pos[known] <= cur_pos[known].unsqueeze(1)

        # ensure at least one feasible action per sample
        none_ok = ~feasible.any(dim=1)
        if none_ok.any():
            feasible[none_ok] = True

        return logits.masked_fill(~feasible, self.invalid_logit_value)

    @staticmethod
    def _constraint_fn(
        kind: str,
        params: Dict[str, Any],
        x_base: torch.Tensor,
        x_derived: torch.Tensor,
        c_base: torch.Tensor,
    ) -> torch.Tensor:
        """Constraint function. All inputs/outputs are (B, 1)."""
        return apply_deterministic_rule(kind, params, x_base, x_derived, c_base)

    def _apply_derived_constraints(
        self,
        gi: int,
        meta: Dict[str, Any],
        x: torch.Tensor,
        c: torch.Tensor,
        c_soft: Optional[torch.Tensor],
        a_apply: torch.Tensor,
        a_apply_soft: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply constraint functions to derived features.

        Called after the base feature has been written into c and c_soft
        returns updated tensors without mutating them in-place.
        """
        derived_list = meta.get("derived")
        rule = meta.get("rule", {})
        if not derived_list or not rule:
            return c, c_soft

        kind = rule.get("kind", "")
        params = rule.get("params", {})
        base_meta_kind = meta["kind"]

        for d_info in derived_list:
            d_kind = d_info["kind"]

            if d_kind == "continuous" and base_meta_kind == "continuous":
                j_base = int(meta["cont_index"])
                j_derived = int(d_info["cont_index"])

                x_base = x[:, j_base : j_base + 1]
                x_derived = x[:, j_derived : j_derived + 1]
                c_base = c[:, j_base : j_base + 1]

                new_derived = self._constraint_fn(
                    kind, params, x_base, x_derived, c_base
                )
                c = self._replace_feature_slice(
                    c,
                    j_derived,
                    j_derived + 1,
                    a_apply * new_derived + (1.0 - a_apply) * x_derived,
                )

                if self.training and c_soft is not None and a_apply_soft is not None:
                    c_soft_base = c_soft[:, j_base : j_base + 1]
                    new_derived_soft = self._constraint_fn(
                        kind, params, x_base, x_derived, c_soft_base
                    )
                    c_soft = self._replace_feature_slice(
                        c_soft,
                        j_derived,
                        j_derived + 1,
                        a_apply_soft * new_derived_soft
                        + (1.0 - a_apply_soft) * x_derived,
                    )

            else:
                raise NotImplementedError(
                    f"Derived constraint not implemented for "
                    f"base kind={base_meta_kind}, derived kind={d_kind} "
                    f"(group '{meta.get('id', '?')}', rule kind='{kind}')."
                )

        return c, c_soft

    @staticmethod
    def _replace_feature_slice(base: torch.Tensor, start: int, end: int, update: torch.Tensor) -> torch.Tensor:
        """Return a tensor with columns [start:end] replaced out-of-place."""
        parts: List[torch.Tensor] = []
        if start > 0:
            parts.append(base[:, :start])
        parts.append(update)
        if end < base.shape[1]:
            parts.append(base[:, end:])
        return torch.cat(parts, dim=1)

    def model_forward(self, x):
        """x is already preprocessed (scaled + one-hot)."""
        if not self._domains_built:
            self._ensure_action_domains()

        z = self.encoder_model(x)
        pred = self.predictor(z)
        y_hat = torch.sigmoid(self.pred_linear(pred))

        policy_input = torch.cat((z, pred), dim=-1)  # z ⊕ p_x

        B = x.shape[0]
        # mask network
        mask_logits = self.mask_head(policy_input).view(B, self.num_groups, 2)
        if self.training:
            # Hard mask for applying actions, with straight-through gradients
            mask_onehot = torch.nn.functional.gumbel_softmax(
                mask_logits, tau=self.gumbel_tau_mask, hard=True, dim=-1
            )
            apply = mask_onehot[..., 1]  # (B, K) {0,1} (ST)

            # Soft mask for proximity loss
            apply_soft = torch.softmax(mask_logits, dim=-1)[..., 1]  # (B, K)
        else:
            probs = torch.softmax(mask_logits, dim=-1)
            apply = (probs[..., 1] > self.mask_threshold).to(z.dtype)
            apply_soft = apply

        # choice network
        choice_logits_all = self.choice_head(policy_input)  # (B, sum A_k)
        c = x.clone()
        c_soft = x.clone() if self.training else None

        # Expected action cost (differentiable) for training-time proximity
        total_cost = torch.zeros((B,), device=x.device) if self.training else None

        for gi, meta in enumerate(self._group_meta):
            a0, a1 = self._action_offsets[gi]
            logits = choice_logits_all[:, a0:a1]
            logits = self._mask_invalid_actions(gi, x, logits)

            if self.training:
                a_onehot = torch.nn.functional.gumbel_softmax(
                    logits, tau=self.gumbel_tau_choice, hard=True, dim=-1
                )
                a_soft = torch.softmax(logits, dim=-1)
            else:
                idx = logits.argmax(dim=-1)
                a_onehot = torch.nn.functional.one_hot(idx, num_classes=logits.shape[-1]).to(logits.dtype)
                a_soft = None

            a_apply = apply[:, gi].unsqueeze(1)  # (B,1)
            a_apply_soft = apply_soft[:, gi].unsqueeze(1) if self.training else None

            # accumulate expected cost
            if self.training and a_soft is not None and total_cost is not None:
                w = float(meta.get("cost_weight", 1.0))
                apply_p = apply_soft[:, gi]  # (B,)
                if meta["kind"] == "continuous":
                    # expected step distance from current value (sentinels treated as OOD)
                    j_cur = int(meta["cont_index"])
                    denom = (self.scaler.max_ - self.scaler.min_)
                    raw_cur = x[:, j_cur] * denom + self.scaler.min_
                    step = float(meta.get("raw_step", 1.0)) or 1.0
                    v0 = float(meta.get("raw_v0", 0.0))
                    idx_cur = torch.round((raw_cur - v0) / step)
                    A = int(meta["action_size"])
                    idx_cur = torch.clamp(idx_cur, 0.0, float(A - 1))
                    # sentinel: virtual index -1 (adds +1 step to all valid moves)
                    for sv in (meta.get("special_values") or []):
                        sv_t = torch.tensor(float(sv), device=raw_cur.device, dtype=raw_cur.dtype)
                        idx_cur = torch.where(raw_cur == sv_t, torch.full_like(idx_cur, -1.0), idx_cur)
                    idx = getattr(self, f"_ag_idx_{gi}")  # (A,)
                    mag = (a_soft * torch.abs(idx.unsqueeze(0) - idx_cur.unsqueeze(1))).sum(dim=1)
                else:
                    (s, e) = meta["slice"]
                    cur_cat = x[:, s:e].argmax(dim=1)
                    pos_map = getattr(self, f"_ag_posmap_{gi}")
                    cur_pos = pos_map[cur_cat]
                    p_cur = torch.zeros((B,), device=x.device, dtype=apply_p.dtype)
                    known = cur_pos >= 0
                    if known.any():
                        p_cur[known] = a_soft[known, cur_pos[known]]
                    mag = 1.0 - p_cur
                total_cost = total_cost + (w * apply_p * (self.action_cost_base + mag))
            if meta["kind"] == "continuous":
                dom = getattr(self, f"_ag_dom_{gi}")  # (A,)
                target = a_onehot @ dom.unsqueeze(1)  # (B,1)
                j = int(meta["cont_index"])
                c = self._replace_feature_slice(
                    c,
                    j,
                    j + 1,
                    a_apply * target + (1.0 - a_apply) * x[:, j : j + 1],
                )
                if self.training and c_soft is not None and a_soft is not None:
                    target_soft = a_soft @ dom.unsqueeze(1)  # (B,1)
                    c_soft = self._replace_feature_slice(
                        c_soft,
                        j,
                        j + 1,
                        a_apply_soft * target_soft + (1.0 - a_apply_soft) * x[:, j : j + 1],
                    )
            else:
                dom_onehot = getattr(self, f"_ag_dom_{gi}")  # (A,m)
                target_block = a_onehot @ dom_onehot  # (B,m)
                (s, e) = meta["slice"]
                c = self._replace_feature_slice(
                    c,
                    s,
                    e,
                    a_apply * target_block + (1.0 - a_apply) * x[:, s:e],
                )
                if self.training and c_soft is not None and a_soft is not None:
                    target_block_soft = a_soft @ dom_onehot  # (B,m)
                    c_soft = self._replace_feature_slice(
                        c_soft,
                        s,
                        e,
                        a_apply_soft * target_block_soft + (1.0 - a_apply_soft) * x[:, s:e],
                    )

            # Type 1/2: apply derived constraints
            if meta.get("derived"):
                c, c_soft = self._apply_derived_constraints(
                    gi, meta, x, c, c_soft, a_apply, a_apply_soft
                )

        if self.training:
            self._last_c_soft = c_soft
            self._last_action_cost = total_cost
        else:
            self._last_c_soft = None
            self._last_action_cost = None

        return torch.squeeze(y_hat, -1), c

__all__ = ['LinearBlock', 'MultilayerPerception', 'BaselinePredictiveModel', 'CounterNetModel', 'CounterNetProjectionModel', 'DiscreteRecourseNetModel']

# Cell
from .import_essentials import *
from .utils import *
from .training_module import PredictiveTrainingModule, CFNetTrainingModule
from .action_groups import (
    ActionGroupSpec,
    apply_deterministic_rule,
)
from .action_projection import (
    ActionProjectionGroup,
    ActionProjectionSpec,
    action_feasibility_mask,
    build_action_projection_spec,
    load_supported_action_groups,
    project_to_actionable,
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
        pred_lr = float(self.hparams.get("predictor_lr", self.lr))
        cf_lr = float(self.hparams.get("cf_lr", self.lr))
        opt_1 = torch.optim.Adam(pred_params, lr=pred_lr)
        opt_2 = torch.optim.Adam(exp_params, lr=cf_lr)
        return (opt_1, opt_2)


class CounterNetProjectionModel(CounterNetModel):
    """CounterNet baseline with inference-time projection onto the actionable set."""

    def __init__(self, config):
        super().__init__(config)
        self.dataset_name: str = str(config.get("dataset_name", ""))
        if not self.dataset_name:
            raise ValueError("CounterNetProjectionModel requires 'dataset_name' in the model config.")
        self._eps: float = float(config.get("mask_eps", 1e-7))

        default_action_groups_path = Path(__file__).resolve().parents[1] / "assets" / "actions" / "action_groups.json"
        self.action_groups_path = Path(config.get("action_groups_path", default_action_groups_path))
        self._action_projection_spec: Optional[ActionProjectionSpec] = None

    def prepare_data(self):
        super().prepare_data()
        self._action_projection_spec = build_action_projection_spec(
            action_groups_path=self.action_groups_path,
            dataset_name=self.dataset_name,
            continous_cols=self.continous_cols,
            discret_cols=self.discret_cols,
            scaler=self.scaler,
            cat_normalizer=self.cat_normalizer,
        )

    def generate_cf(self, x, clamp=False):
        self.freeze()
        if self._action_projection_spec is None:
            raise RuntimeError("CounterNetProjectionModel projection spec has not been prepared. Call prepare_data() first.")

        _, c = self.model_forward(x)
        cf_soft = self.cat_normalizer.normalize(c, hard=False)
        if clamp and self.cat_normalizer.cat_idx > 0:
            cf_soft = cf_soft.clone()
            cf_soft[:, : self.cat_normalizer.cat_idx] = torch.clamp(
                cf_soft[:, : self.cat_normalizer.cat_idx],
                0.0,
                1.0,
            )
        return project_to_actionable(x, cf_soft, self._action_projection_spec, eps=self._eps)


class DiscreteRecourseNetModel(CFNetTrainingModule):
    """DiscreteRecourseNet: a generator of actionable counterfactuals, 
    with gumbel-softmax straight-through sampling on mask and choice networks, 
    and invalid action masking for monotonicity constraints.

    Currently implemented:
      - Type 0 (singleton) action groups
      - Type 1/2 (base + derived) action groups
      - Type 4 (joint latent-shift) action groups

    Expected action_groups.json schema:
      action_groups[dataset][group_id] = {
        "type": 0|1|2|4,
        "features": ["feature_name", ...],
        "mutable": true/false,
        "action_domain": {"kind": "values"|"noop", "feature": "...", "values": [...]} |
                         {"kind": "delta_steps", "deltas": [...], "scales": {...}, "domains": {...}},
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
      - Type 4 groups act on all listed continuous features using one shared discrete delta.
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
        self._eps: float = float(config.get("mask_eps", 1e-7))

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

        # Built after prepare_data: shared actionable runtime metadata
        self._domains_built: bool = False
        self._sanity_logged: bool = False
        self._action_runtime: Optional[ActionProjectionSpec] = None
        self._group_cost_weights: List[float] = []

        # Soft (expected) counterfactual used for proximity loss during training
        self._last_c_soft: Optional[torch.Tensor] = None

        # Expected (differentiable) action cost for training-time proximity.
        self.action_cost_base: float = float(config.get('action_cost_base'))
        self.action_cost_weights: Dict[str, float] = dict(config.get('action_cost_weights', {}) or {})
        self._last_action_cost: Optional[torch.Tensor] = None


    # Action group parsing/build
    def _parse_action_groups(self) -> None:
        group_specs, inc_only, dec_only = load_supported_action_groups(
            self.action_groups_path,
            self.dataset_name,
            learnable_only=True,
        )
        self._group_specs = list(group_specs)
        self._inc_only = set(inc_only)
        self._dec_only = set(dec_only)

    def _default_cost_weight(self, gid: str, feat: str, kind: str, action_size: int, action_kind: str) -> float:
        # can be overridden by config
        if gid in self.action_cost_weights:
            return float(self.action_cost_weights[gid])
        if feat in self.action_cost_weights:
            return float(self.action_cost_weights[feat])

        if action_kind == "delta_steps":
            return 1.0

        # normalize continuous domains by number of discrete steps
        if kind == "continuous":
            return 1.0 / max(1, int(action_size) - 1)
        # categorical: normalize by number of categories
        return 1.0 / max(1, int(action_size) - 1)

    def prepare_data(self):
        super().prepare_data()
        self._ensure_action_runtime()

    def _ensure_action_runtime(self) -> None:
        if self._domains_built:
            return

        self._action_runtime = build_action_projection_spec(
            action_groups_path=self.action_groups_path,
            dataset_name=self.dataset_name,
            continous_cols=self.continous_cols,
            discret_cols=self.discret_cols,
            scaler=self.scaler,
            cat_normalizer=self.cat_normalizer,
            group_specs=self._group_specs,
            increase_only=self._inc_only,
            decrease_only=self._dec_only,
        )
        runtime_groups = list(self._action_runtime.groups)
        self._group_cost_weights = [
            self._default_cost_weight(
                group.id,
                str(group.base_feature),
                group.kind,
                int(group.action_size),
                str(group.action_kind),
            )
            for group in runtime_groups
        ]

        if len(runtime_groups) != self.num_groups:
            raise RuntimeError(
                f"Action runtime built {len(runtime_groups)} groups, but the model expects {self.num_groups}."
            )

        runtime_action_sizes = [int(group.action_size) for group in runtime_groups]
        if runtime_action_sizes != self._action_sizes:
            raise RuntimeError(
                f"Action runtime sizes {runtime_action_sizes} do not match initialized sizes {self._action_sizes}."
            )

        # group composition printout (continuous vs categorical)
        if not self._sanity_logged:
            n_cont = sum(1 for group in runtime_groups if group.kind == "continuous")
            n_cat = sum(1 for group in runtime_groups if group.kind == "categorical")
            n_t0 = sum(1 for group in runtime_groups if int(group.type) == 0)
            n_t12 = sum(1 for group in runtime_groups if int(group.type) in (1, 2))
            n_t4 = sum(1 for group in runtime_groups if int(group.type) == 4)
            print(
                f"[DiscreteRecourseNet] Loaded {len(runtime_groups)} groups "
                f"(type0={n_t0}, type1+2={n_t12}, type4={n_t4}, "
                f"continuous={n_cont}, categorical={n_cat}) "
                f"for dataset='{self.dataset_name}'."
            )
            if n_cat == 0 and len(self.discret_cols) > 0:
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
        pred_lr = float(self.hparams.get("predictor_lr", self.lr))
        cf_lr = float(self.hparams.get("cf_lr", self.lr))
        opt_1 = torch.optim.Adam(pred_params, lr=pred_lr)
        opt_2 = torch.optim.Adam(policy_params, lr=cf_lr)
        return (opt_1, opt_2)

    # Invalid action masking for monotonicity constraints
    def _mask_invalid_actions(self, group: ActionProjectionGroup, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Mask infeasible actions for one group before sampling/argmax."""
        feasible = action_feasibility_mask(
            group,
            x,
            eps=self._eps,
            ensure_any=True,
        )
        if feasible.numel() == 0:
            return logits
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
        group: ActionProjectionGroup,
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
        derived_list = list(group.derived)
        rule = dict(group.rule)
        if not derived_list or not rule:
            return c, c_soft

        kind = rule.get("kind", "")
        params = rule.get("params", {})
        base_meta_kind = group.kind

        for d_info in derived_list:
            d_kind = d_info.kind

            if d_kind == "continuous" and base_meta_kind == "continuous":
                if group.cont_index is None or d_info.cont_index is None:
                    raise ValueError(f"Derived constraint group '{group.id}' is missing continuous indices.")
                j_base = int(group.cont_index)
                j_derived = int(d_info.cont_index)

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
                    f"(group '{group.id}', rule kind='{kind}')."
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
            self._ensure_action_runtime()
        if self._action_runtime is None:
            raise RuntimeError("Action runtime has not been prepared.")

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

        runtime_groups = self._action_runtime.groups
        scaler_min = float(self._action_runtime.scaler_min)
        scaler_max = float(self._action_runtime.scaler_max)
        denom = float(scaler_max - scaler_min) if float(scaler_max - scaler_min) != 0.0 else 1.0

        for gi, group in enumerate(runtime_groups):
            a0, a1 = self._action_offsets[gi]
            logits = choice_logits_all[:, a0:a1]
            logits = self._mask_invalid_actions(group, x, logits)

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
                w = float(self._group_cost_weights[gi])
                apply_p = apply_soft[:, gi]  # (B,)
                if group.action_kind == "delta_steps":
                    delta_mag = group.delta_domain.to(device=x.device, dtype=apply_p.dtype).abs()
                    mag = (a_soft * delta_mag.unsqueeze(0)).sum(dim=1)
                elif group.kind == "continuous":
                    # expected step distance from current value (sentinels treated as OOD)
                    if group.cont_index is None:
                        raise ValueError(f"Continuous group '{group.id}' is missing cont_index.")
                    j_cur = int(group.cont_index)
                    raw_cur = x[:, j_cur] * float(denom) + float(scaler_min)
                    step = float(group.raw_step or 1.0) or 1.0
                    v0 = float(group.raw_v0 or 0.0)
                    idx_cur = torch.round((raw_cur - v0) / step)
                    A = int(group.action_size)
                    idx_cur = torch.clamp(idx_cur, 0.0, float(A - 1))
                    # sentinel: virtual index -1 (adds +1 step to all valid moves)
                    for sv in (group.special_values or []):
                        sv_t = torch.tensor(float(sv), device=raw_cur.device, dtype=raw_cur.dtype)
                        idx_cur = torch.where(raw_cur == sv_t, torch.full_like(idx_cur, -1.0), idx_cur)
                    idx = torch.arange(A, device=x.device, dtype=apply_p.dtype)
                    mag = (a_soft * torch.abs(idx.unsqueeze(0) - idx_cur.unsqueeze(1))).sum(dim=1)
                else:
                    if group.slice is None:
                        raise ValueError(f"Categorical group '{group.id}' is missing slice.")
                    (s, e) = group.slice
                    cur_cat = x[:, s:e].argmax(dim=1)
                    pos_map = group.pos_map.to(device=x.device)
                    cur_pos = pos_map[cur_cat]
                    p_cur = torch.zeros((B,), device=x.device, dtype=apply_p.dtype)
                    known = cur_pos >= 0
                    if known.any():
                        p_cur[known] = a_soft[known, cur_pos[known]]
                    mag = 1.0 - p_cur
                total_cost = total_cost + (w * apply_p * (self.action_cost_base + mag))
            if group.action_kind == "delta_steps":
                if not group.cont_indices:
                    raise ValueError(f"Type 4 group '{group.id}' is missing cont_indices.")
                shift = a_onehot @ group.scaled_shifts.to(device=x.device, dtype=x.dtype)  # (B,F)
                x_block = torch.stack([x[:, idx] for idx in group.cont_indices], dim=1)
                target_block = x_block + shift
                updated_block = a_apply * target_block + (1.0 - a_apply) * x_block
                for pos, j in enumerate(group.cont_indices):
                    c = self._replace_feature_slice(
                        c,
                        j,
                        j + 1,
                        updated_block[:, pos : pos + 1],
                    )

                if self.training and c_soft is not None and a_soft is not None:
                    shift_soft = a_soft @ group.scaled_shifts.to(device=x.device, dtype=x.dtype)
                    target_block_soft = x_block + shift_soft
                    updated_block_soft = a_apply_soft * target_block_soft + (1.0 - a_apply_soft) * x_block
                    for pos, j in enumerate(group.cont_indices):
                        c_soft = self._replace_feature_slice(
                            c_soft,
                            j,
                            j + 1,
                            updated_block_soft[:, pos : pos + 1],
                        )

            elif group.kind == "continuous":
                if group.cont_index is None:
                    raise ValueError(f"Continuous group '{group.id}' is missing cont_index.")
                dom = group.scaled_domain.to(device=x.device, dtype=x.dtype)  # (A,)
                target = a_onehot @ dom.unsqueeze(1)  # (B,1)
                j = int(group.cont_index)
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
                if group.slice is None:
                    raise ValueError(f"Categorical group '{group.id}' is missing slice.")
                dom_onehot = group.scaled_domain.to(device=x.device, dtype=x.dtype)  # (A,m)
                target_block = a_onehot @ dom_onehot  # (B,m)
                (s, e) = group.slice
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
            if group.derived:
                c, c_soft = self._apply_derived_constraints(
                    group, x, c, c_soft, a_apply, a_apply_soft
                )

        if self.training:
            self._last_c_soft = c_soft
            self._last_action_cost = total_cost
        else:
            self._last_c_soft = None
            self._last_action_cost = None

        return torch.squeeze(y_hat, -1), c

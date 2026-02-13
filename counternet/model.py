__all__ = ['LinearBlock', 'MultilayerPerception', 'BaselinePredictiveModel', 'CounterNetModel', 'DiscreteRecourseNetModel']

# Cell
from .import_essentials import *
from .utils import *
from .training_module import BaseModule, PredictiveTrainingModule, CFNetTrainingModule

import json
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
        exp_dims = [x for x in self.exp_dims]
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


class DiscreteRecourseNetModel(CFNetTrainingModule):
    """two-stage discrete counterfactual generator

    - Type 0 action groups
    - Gumbel-Softmax straight-through sampling (mask + choice networks)
    - Invalid action masking for monotonicity (increase-only / decrease-only)

    Expected action_groups.json schema:
      action_groups[dataset][group_id] = {
        "type": 0,
        "features": ["feature_name"],   # singleton
        "mutable": true/false,
        "action_domain": {"kind": "values"|"noop", "feature": "...", "values": [...]},
      }
      action_groups[dataset]["__constraints__"]["monotonicity"] = {
        "increase_only": [...], "decrease_only": [...]
      }

    Notes:
      - Domains in action_groups.json are in feature space (raw values for continuous, category labels for discrete).
      - This model maps those domains into the preprocessed input space using the fitted MinMaxScaler and OneHotEncoder
        categories stored in CategoricalNormalizer.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assert self.enc_dims[-1] == self.dec_dims[0]

        # networks: encoder + predictor
        self.encoder_model = MultilayerPerception(self.enc_dims)
        self.predictor = MultilayerPerception(self.dec_dims)
        self.pred_linear = nn.Linear(self.dec_dims[-1], 1)

        # discrete policy hyperparams
        self.gumbel_tau_mask: float = float(config.get("gumbel_tau_mask", 1.0))
        self.gumbel_tau_choice: float = float(config.get("gumbel_tau_choice", 1.0))
        self.mask_threshold: float = float(config.get("mask_threshold", 0.5))
        self.invalid_logit_value: float = float(config.get("invalid_logit_value", -1e9))
        self._eps: float = float(config.get("mask_eps", 1e-6))

        # load groups from action_groups.json
        self.dataset_name: str = str(config.get("dataset_name", ""))
        self.action_groups_path: Path = Path(config.get("action_groups_path", "assets/actions/action_groups.json"))

        self._group_specs: List[Dict[str, Any]] = []
        self._inc_only: set[str] = set()
        self._dec_only: set[str] = set()

        self._parse_action_groups()

        self.num_groups: int = len(self._group_specs)
        if self.num_groups == 0:
            raise ValueError(
                f"No Type 0 mutable groups found for dataset='{self.dataset_name}' in {self.action_groups_path}. "
                "Run generate_action_groups.py to produce a full action_groups.json with Type 0 singleton groups."
            )

        # action space over groups
        self._action_sizes: List[int] = [int(g["action_size"]) for g in self._group_specs]
        self._action_offsets: List[Tuple[int, int]] = []
        offset = 0
        for a in self._action_sizes:
            self._action_offsets.append((offset, offset + a))
            offset += a
        self.total_actions: int = offset

        z_dim = self.enc_dims[-1]
        # 2-class per group (apply vs not apply)
        self.mask_head = nn.Linear(z_dim, self.num_groups * 2)
        # logits for each group's action domain
        self.choice_head = nn.Linear(z_dim, self.total_actions)

        self._domains_built: bool = False
        self._sanity_logged: bool = False

    def _parse_action_groups(self) -> None:
        if not self.action_groups_path.exists():
            raise FileNotFoundError(f"action_groups.json not found at: {self.action_groups_path}")

        with self.action_groups_path.open("r") as f:
            ag = json.load(f)

        ds = ag.get(self.dataset_name, {})
        if not isinstance(ds, dict):
            raise ValueError(f"Invalid action_groups for dataset '{self.dataset_name}'")

        constraints = ds.get("__constraints__", {}) if isinstance(ds.get("__constraints__", {}), dict) else {}
        mono = constraints.get("monotonicity", {}) if isinstance(constraints.get("monotonicity", {}), dict) else {}
        self._inc_only = set(mono.get("increase_only", []) or [])
        self._dec_only = set(mono.get("decrease_only", []) or [])

        # order for reproducibility
        for gid in sorted([k for k in ds.keys() if k != "__constraints__"]):
            g = ds[gid]
            if not isinstance(g, dict):
                continue
            if int(g.get("type", -1)) != 0:
                continue  # only Type 0 for now

            feats = g.get("features")
            if not feats:
                ad = g.get("action_domain", {}) if isinstance(g.get("action_domain", {}), dict) else {}
                if ad.get("feature") is not None:
                    feats = [ad.get("feature")]
            if not feats or len(feats) != 1:
                continue
            feat = str(feats[0])

            mutable = bool(g.get("mutable", True))
            ad = g.get("action_domain", {}) if isinstance(g.get("action_domain", {}), dict) else {}
            kind = str(ad.get("kind", "values"))
            vals = ad.get("values", []) if kind == "values" else []

            # ignore immutable and noop groups for the learnable action policy
            if (not mutable) or kind == "noop":
                continue
            if kind != "values":
                continue
            if not isinstance(vals, list) or len(vals) == 0:
                continue

            self._group_specs.append(
                {
                    "id": gid,
                    "feature": feat,
                    "action_values": vals,
                    "action_size": len(vals),
                    "increase_only": feat in self._inc_only,
                    "decrease_only": feat in self._dec_only,
                }
            )

    def prepare_data(self):
        super().prepare_data()
        self._ensure_action_domains()

    def _ensure_action_domains(self) -> None:
        """Build tensors in preprocessed input space.

        Continuous:
          - store (A,) scaled target values using fitted MinMaxScaler
        Categorical:
          - store (A, m) one-hot target blocks using fitted OneHotEncoder categories
          - store (m,) map from category index -> position in domain list (for monotonic rank masking)
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
        for gi, g in enumerate(self._group_specs):
            feat = g["feature"]
            vals = g["action_values"]

            if feat in cont_pos:
                j = int(cont_pos[feat])

                raw = torch.tensor([float(v) for v in vals], dtype=torch.float32)
                denom = (self.scaler.max_ - self.scaler.min_)
                if float(denom) == 0.0:
                    denom = torch.tensor(1.0, dtype=torch.float32)
                dom = (raw - self.scaler.min_) / denom
                self.register_buffer(f"_ag_dom_{gi}", dom)
                self._group_meta.append(
                    {
                        "id": g["id"],
                        "feature": feat,
                        "kind": "continuous",
                        "cont_index": j,
                        "action_size": int(g["action_size"]),
                        "increase_only": bool(g.get("increase_only", False)),
                        "decrease_only": bool(g.get("decrease_only", False)),
                    }
                )

            elif feat in disc_pos:
                di = int(disc_pos[feat])
                if di >= len(categories) or di >= len(cat_slices):
                    raise ValueError(f"Discrete feature '{feat}' missing category metadata.")

                cats = list(categories[di])
                cat_to_index = {c: j for j, c in enumerate(cats)}

                domain_cat_indices: List[int] = []
                for v in vals:
                    if v in cat_to_index:
                        domain_cat_indices.append(int(cat_to_index[v]))
                    else:
                        found = False
                        for c, j in cat_to_index.items():
                            try:
                                if float(c) == float(v):
                                    domain_cat_indices.append(int(j))
                                    found = True
                                    break
                            except Exception:
                                continue
                        if not found:
                            raise ValueError(
                                f"Domain value '{v}' for feature '{feat}' not found in fitted categories."
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
                self._group_meta.append(
                    {
                        "id": g["id"],
                        "feature": feat,
                        "kind": "categorical",
                        "slice": (int(s), int(e)),
                        "action_size": int(g["action_size"]),
                        "num_categories": int(m),
                        "increase_only": bool(g.get("increase_only", False)),
                        "decrease_only": bool(g.get("decrease_only", False)),
                    }
                )
            else:
                raise ValueError(
                    f"Feature '{feat}' (group '{g['id']}') not found in continous_cols or discret_cols."
                )

        # Log loaded groups
        if not self._sanity_logged:
            n_cont = sum(1 for mm in self._group_meta if mm.get("kind") == "continuous")
            n_cat = sum(1 for mm in self._group_meta if mm.get("kind") == "categorical")
            print(
                f"[DiscreteRecourseNet] Loaded {len(self._group_meta)} Type-0 groups "
                f"(continuous={n_cont}, categorical={n_cat}) for dataset='{self.dataset_name}'."
            )
            if n_cat == 0 and len(disc_cols) > 0:
                print(
                    "[DiscreteRecourseNet][WARN] No categorical Type-0 groups were loaded even though "
                    "the dataset has discrete columns. Check action_groups.json domains / feature names."
                )
            self._sanity_logged = True

        self._domains_built = True

    def forward(self, x, hard: bool = False):
        if self.training:
            hard = True
        return super().forward(x, hard=hard)

    def _mask_invalid_actions(self, gi: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Instance-dependent invalid action masking"""
        meta = self._group_meta[gi]
        inc_only = bool(meta.get("increase_only", False))
        dec_only = bool(meta.get("decrease_only", False))
        if not (inc_only or dec_only):
            return logits

        A = logits.shape[-1]
        device = logits.device

        if meta["kind"] == "continuous":
            dom = getattr(self, f"_ag_dom_{gi}")
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

    def model_forward(self, x):
        """x is already preprocessed (scaled + one-hot)"""
        if not self._domains_built:
            self._ensure_action_domains()

        z = self.encoder_model(x)
        pred = self.predictor(z)
        y_hat = torch.sigmoid(self.pred_linear(pred))

        B = x.shape[0]
        # mask network
        mask_logits = self.mask_head(z).view(B, self.num_groups, 2)
        if self.training:
            mask_onehot = torch.nn.functional.gumbel_softmax(mask_logits, tau=self.gumbel_tau_mask, hard=True, dim=-1)
            apply = mask_onehot[..., 1]  # (B, K)
        else:
            probs = torch.softmax(mask_logits, dim=-1)
            apply = (probs[..., 1] > self.mask_threshold).to(z.dtype)

        # choice network
        choice_logits_all = self.choice_head(z)  # (B, sum A_k)
        c = x.clone()

        for gi, meta in enumerate(self._group_meta):
            a0, a1 = self._action_offsets[gi]
            logits = choice_logits_all[:, a0:a1]
            logits = self._mask_invalid_actions(gi, x, logits)

            if self.training:
                a_onehot = torch.nn.functional.gumbel_softmax(logits, tau=self.gumbel_tau_choice, hard=True, dim=-1)
            else:
                idx = logits.argmax(dim=-1)
                a_onehot = torch.nn.functional.one_hot(idx, num_classes=logits.shape[-1]).to(logits.dtype)

            a_apply = apply[:, gi].unsqueeze(1)  # (B,1)

            if meta["kind"] == "continuous":
                dom = getattr(self, f"_ag_dom_{gi}")  # (A,)
                target = a_onehot @ dom.unsqueeze(1)  # (B,1)
                j = int(meta["cont_index"])
                c[:, j : j + 1] = a_apply * target + (1.0 - a_apply) * x[:, j : j + 1]
            else:
                dom_onehot = getattr(self, f"_ag_dom_{gi}")  # (A,m)
                target_block = a_onehot @ dom_onehot  # (B,m)
                (s, e) = meta["slice"]
                c[:, s:e] = a_apply * target_block + (1.0 - a_apply) * x[:, s:e]

        return torch.squeeze(y_hat, -1), c

__all__ = ['BaseModule', 'PredictiveTrainingModule', 'CFNetTrainingModule']


from .import_essentials import *
from .utils import *
from .evaluation import SensitivityMetric, proximity
from .base_interface import ABCBaseModule, GlobalExplainerBase


class BaseModule(pl.LightningModule, ABCBaseModule):
    def __init__(self, configs: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(configs)

        # read data
        self.data = pd.read_csv(Path(configs['data_dir']))
        self.continous_cols = configs['continous_cols']
        self.discret_cols = configs['discret_cols']
        self.__check_cols()

        # set training configs
        self.lr = configs['lr']
        self.batch_size = configs['batch_size']
        self.dropout = configs['dropout'] if 'dropout' in configs.keys() else 0.3
        self.lambda_1 = configs['lambda_1'] if 'lambda_1' in configs.keys() else 1
        self.lambda_2 = configs['lambda_2'] if 'lambda_2' in configs.keys() else 1
        self.lambda_3 = configs['lambda_3'] if 'lambda_3' in configs.keys() else 1
        self.threshold = configs['threshold'] if 'threshold' in configs.keys() else 1
        self.smooth_y = configs['smooth_y'] if 'smooth_y' in configs.keys() else True
        self.prediction_threshold_mode = str(configs.get('prediction_threshold_mode', 'round'))
        self.prediction_threshold = float(configs.get('prediction_threshold', 0.5))
        self._resolved_prediction_threshold: Optional[float] = None

        supported_threshold_modes = {'round', 'fixed', 'auto_val_balanced_accuracy'}
        if self.prediction_threshold_mode not in supported_threshold_modes:
            raise ValueError(
                f"Unsupported prediction_threshold_mode '{self.prediction_threshold_mode}'. "
                f"Expected one of {sorted(supported_threshold_modes)}."
            )

        # loss functions
        self.loss_func_1 = get_loss_functions(configs['loss_1']) if 'loss_1' in configs.keys() else get_loss_functions("mse")
        self.loss_func_2 = get_loss_functions(configs['loss_2']) if 'loss_2' in configs.keys() else get_loss_functions("mse")
        self.loss_func_3 = get_loss_functions(configs['loss_3']) if 'loss_3' in configs.keys() else get_loss_functions("mse")

        # set model configss
        self.enc_dims = configs['encoder_dims'] if 'encoder_dims' in configs.keys() else []
        self.dec_dims = configs['decoder_dims'] if 'decoder_dims' in configs.keys() else []
        self.exp_dims = configs['explainer_dims'] if 'explainer_dims' in configs.keys() else []

        # log graph
        self.example_input_array = torch.randn((1, self.enc_dims[0]))

    def __check_cols(self):
        assert sorted(list(self.data.columns[:-1])) == sorted(self.continous_cols + self.discret_cols), \
            f"data columns ({sorted(list(self.data.columns[:-1]))}) is not the same as continous_cols and discret_cols ({sorted(self.continous_cols + self.discret_cols)})"
        self.data = self.data.astype(
            {col: float for col in self.continous_cols})

    def __check_cat_size(self, X_cat: torch.Tensor, categories: List[List[Any]]):
        n = 0
        for cat in categories:
            n += len(cat)
        assert X_cat.size(-1) == n

    def training_epoch_end(self, outs):
        if self.current_epoch == 0:
            self.logger.log_hyperparams(self.hparams)

    def prepare_data(self):
        # TODO Decouple data preparision and use `LightningDataModule`
        # 70% for training, 10% for validation, 20% for testing
        X, y = split_X_y(self.data)

        # preprocessing
        self.scaler = MinMaxScaler()
        self.ohe = OneHotEncoder()
        X_cont = self.scaler.fit_transform(X[self.continous_cols]) if self.continous_cols else torch.tensor([[] for _ in range(len(X))])
        X_cat = self.ohe.fit_transform(X[self.discret_cols]) if self.discret_cols else torch.tensor([[] for _ in range(len(X))])
        X = torch.cat((X_cont, X_cat), dim=1)

        # init categorical normalizer to enable categorical features to be one-hot-encoding format
        cat_arrays = self.ohe.categories_ if self.discret_cols else []
        self.cat_normalizer = CategoricalNormalizer(cat_arrays, cat_idx=len(self.continous_cols))
        self.__check_cat_size(X_cat, cat_arrays)

        # init sensitivity metric
        self.sensitivity = SensitivityMetric(
            predict_fn=self.predict,
            scaler=self.scaler,
            cat_idx=len(self.continous_cols),
            threshold=self.threshold,
            label_fn=self.binarize_prediction_scores,
        )

        print(f"x_cont: {X_cont.size()}, x_cat: {X_cat.size()}, X shape: {X.size()}")

        assert X.size(-1) == self.enc_dims[0],\
            f'The input dimension X (shape: {X.shape[-1]})  != encoder_dims[0]: {self.enc_dims}'

        # prepare train & test
        train, val, test = train_val_test_split(X, y)
        self.train_dataset = TensorDataset(*train)
        self.val_dataset = TensorDataset(*val)
        self.test_dataset = TensorDataset(*test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          pin_memory=True, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          pin_memory=True, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          pin_memory=True, shuffle=False, num_workers=0)

    def current_prediction_threshold(self) -> float:
        if self.prediction_threshold_mode == 'round':
            return float(self.prediction_threshold)
        if self._resolved_prediction_threshold is not None:
            return float(self._resolved_prediction_threshold)
        return float(self.prediction_threshold)

    def binarize_prediction_scores(self, scores: torch.Tensor) -> torch.Tensor:
        return binarize_binary(
            scores,
            threshold=self.current_prediction_threshold(),
            mode=self.prediction_threshold_mode,
        )

    def flip_prediction_scores(self, scores: torch.Tensor) -> torch.Tensor:
        return flip_binary(
            scores,
            threshold=self.current_prediction_threshold(),
            mode=self.prediction_threshold_mode,
        )

    def resolve_prediction_threshold(self, force: bool = False) -> Optional[float]:
        if self.prediction_threshold_mode == 'round':
            self._resolved_prediction_threshold = None
            return None

        if self.prediction_threshold_mode == 'fixed':
            self._resolved_prediction_threshold = float(self.prediction_threshold)
            return float(self._resolved_prediction_threshold)

        if (not force) and (self._resolved_prediction_threshold is not None):
            return float(self._resolved_prediction_threshold)

        if not hasattr(self, 'val_dataset'):
            raise RuntimeError("Validation dataset is not prepared. Call prepare_data() before resolving thresholds.")

        param = next(self.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        was_training = self.training
        self.eval()

        scores: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        loader = self.val_dataloader()

        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(device)
                scores.append(self.predict_proba(x).detach().cpu())
                targets.append(y.detach().cpu())

        if was_training:
            self.train()

        if scores:
            threshold = select_balanced_accuracy_threshold(
                torch.cat(scores, dim=0),
                torch.cat(targets, dim=0),
                default=self.prediction_threshold,
            )
        else:
            threshold = float(self.prediction_threshold)

        self._resolved_prediction_threshold = float(threshold)
        return float(self._resolved_prediction_threshold)


class PredictiveTrainingModule(BaseModule):
    def __init__(self, configs: Dict[str, Any]):
        super().__init__(configs)
        # define metrics
        self.val_acc = Accuracy()

    def forward(self, *x):
        return self.model_forward(x)

    def predict_proba(self, x):
        return self(x)

    def predict(self, x):
        y_hat = self.predict_proba(x)
        return self.binarize_prediction_scores(y_hat)

    def configure_optimizers(self):
        pred_lr = float(self.hparams.get('predictor_lr', self.lr))
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=pred_lr)

    def training_step(self, batch, batch_idx):
        # batch
        *x, y = batch
        # fwd
        y_hat = self(*x)
        # loss
        if self.smooth_y:
            y = smooth_y(y)
        loss = F.binary_cross_entropy(y_hat, y)

        # Logging to TensorBoard
        self.log('train/train_loss_1', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        *x, y = batch
        # fwd
        y_hat = self(*x)
        # loss
        loss = F.binary_cross_entropy(y_hat, y)

        self.log('val/val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val/pred_accuracy', self.val_acc(y_hat, y.int()), on_step=False, on_epoch=True, sync_dist=True)


class CFNetTrainingModule(BaseModule, GlobalExplainerBase):
    def __init__(self, configs: Dict[str, Any]):
        super().__init__(configs)
        # define metrics
        self.pred_acc = Accuracy()
        self.cf_acc = Accuracy()
        # self.proximity = ProximityMetric()

        # "all" or "neg_only" (only y==0 used to train generator)
        self.cf_target_filter = configs.get("cf_target_filter", "all")

        # L2 printouts:
        self.l2_ema_beta = float(configs.get("l2_ema_beta", 0.9))
        self._l2_ema_epoch = -1
        self._l2_ema_ratio = None
        self._l2_ema_l2 = None
        self._l2_ema_l3 = None

    def _ensure_l2_ema_epoch(self):
        ep = int(self.current_epoch)
        if getattr(self, "_l2_ema_epoch", -1) != ep:
            self._l2_ema_epoch = ep
            self._l2_ema_ratio = None
            self._l2_ema_l2 = None
            self._l2_ema_l3 = None

    def _update_l2_tuning_ema(self, l_2: torch.Tensor, l_3: torch.Tensor):
        if self.trainer is not None and not getattr(self.trainer, "is_global_zero", True):
            return
        self._ensure_l2_ema_epoch()
        with torch.no_grad():
            eps = 1e-8
            l2 = float(l_2.detach().item())
            l3 = float(l_3.detach().item())
            ratio = (float(self.lambda_2) * l2) / (float(self.lambda_3) * l3 + eps)

            beta = float(getattr(self, "l2_ema_beta", 0.9))

            def ema(prev, val):
                return val if prev is None else (beta * prev + (1.0 - beta) * val)

            self._l2_ema_ratio = ema(self._l2_ema_ratio, ratio)
            self._l2_ema_l2 = ema(self._l2_ema_l2, l2)
            self._l2_ema_l3 = ema(self._l2_ema_l3, l3)

    def training_epoch_end(self, outs):
        # hyperparams logging
        super().training_epoch_end(outs)

        if self.trainer is not None and not getattr(self.trainer, "is_global_zero", True):
            return
        if getattr(self, "_l2_ema_ratio", None) is None:
            return

        ratio = float(self._l2_ema_ratio)

        self.print(
            f"[l2-tune][epoch] epoch={int(self.current_epoch)} "
            f"ema_l2={float(self._l2_ema_l2):.4f} ema_l3={float(self._l2_ema_l3):.4f} "
            f"ema_ratio=(λ2·l2)/(λ3·l3)={ratio:.3f}"
        )

    def forward(self, x, hard: bool=False):
        """hard: categorical features in counterfactual is one-hot-encoding or not"""
        y, c = self.model_forward(x)
        c = self.cat_normalizer.normalize(c, hard=hard)
        return y, c

    def predict_proba(self, x):
        y_hat, _ = self.model_forward(x)
        return y_hat

    def predict(self, x):
        """x has not been preprocessed"""
        y_hat = self.predict_proba(x)
        return self.binarize_prediction_scores(y_hat)

    def generate_cf(self, x, clamp=False):
        self.freeze()
        y, c = self.model_forward(x)
        if clamp:
            c = torch.clamp(c, 0., 1.)
        return self.cat_normalizer.normalize(c, hard=True)

    def _logging_loss(self, *loss, stage: str, on_step: bool = False):
        for i, l in enumerate(loss):
            self.log(f'{stage}/{stage}_loss_{i+1}', l, on_step=on_step, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def _loss_functions(self, x, c, y, y_hat, y_prime=None, is_val=False):
        """
        x: input value
        c: conterfactual example
        y: ground truth
        y_hat: predicted result
        y_prime_mode: 'label' or 'predicted'
        """
        if y_prime is None:
            y_prime = self.flip_prediction_scores(y_hat)

        # Predict label for c
        if hasattr(self, "predict_proba") and callable(getattr(self, "predict_proba")):
            c_y = self.predict_proba(c)
        else:
            c_y, _ = self.model_forward(c)

        if self.smooth_y and not is_val:
            y = smooth_y(y)
            y_prime = smooth_y(y_prime)

        l_1 = self.loss_func_1(y_hat, y)

        # Proximity term:
        action_cost = getattr(self, "_action_cost_override", None)
        if action_cost is None and not is_val:
            action_cost = getattr(self, "_last_action_cost", None)

        if (not is_val) and isinstance(action_cost, torch.Tensor) and action_cost.ndim == 1 and action_cost.shape[0] == x.shape[0]:
            l_2 = action_cost.mean()
        else:
            prox_cf = getattr(self, '_prox_override', None)
            if (not is_val) and isinstance(prox_cf, torch.Tensor) and prox_cf.shape == c.shape:
                l_2 = self.loss_func_2(x, prox_cf)
            else:
                l_2 = self.loss_func_2(x, c)

        l_3 = self.loss_func_3(c_y, y_prime)

        return l_1, l_2, l_3

    def configure_optimizers(self):
        pred_lr = float(self.hparams.get('predictor_lr', self.lr))
        cf_lr = float(self.hparams.get('cf_lr', self.lr))
        opt_1 = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=pred_lr)
        opt_2 = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=cf_lr)
        return (opt_1, opt_2)

    def predictor_step(self, l_1, l_3):
        p_loss = self.lambda_1 * l_1 # + self.lambda_3 * l_3
        self.log('train/p_loss', p_loss, on_step=False, on_epoch=True, sync_dist=True)
        return p_loss

    def explainer_step(self, l_2, l_3):
        e_loss = self.lambda_2 * l_2 + self.lambda_3 * l_3
        self.log('train/e_loss', e_loss, on_step=False, on_epoch=True, sync_dist=True)
        return e_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        y_hat, c = self(x)

        # Soft CF for proximity loss:  used for L2
        prox_c = getattr(self, '_last_c_soft', None)
        if (not isinstance(prox_c, torch.Tensor)) or (prox_c.shape != c.shape):
            prox_c = None

        # Optional action-cost vector (DiscreteRecourseNet); shape [B].
        action_cost = getattr(self, "_last_action_cost", None)
        if (not isinstance(action_cost, torch.Tensor)) or (action_cost.ndim != 1) or (action_cost.shape[0] != x.shape[0]):
            action_cost = None

        def call_loss(x_, c_, y_, y_hat_, **kwargs):
            prox_override = kwargs.pop('_prox_override', None)
            if prox_override is None:
                prox_override = prox_c

            action_cost_override = kwargs.pop("_action_cost_override", None)
            if action_cost_override is None:
                action_cost_override = action_cost

            if prox_override is not None:
                self._prox_override = prox_override
            if action_cost_override is not None:
                self._action_cost_override = action_cost_override
            try:
                return self._loss_functions(x_, c_, y_, y_hat_, **kwargs)
            finally:
                if hasattr(self, '_prox_override'):
                    delattr(self, '_prox_override')
                if hasattr(self, '_action_cost_override'):
                    delattr(self, '_action_cost_override')

        # Ablation modes for CF targets:
        # "all": optimize generator on all samples
        # "neg_only": optimize generator only on negatives (y==0)
        # "validity_neg_only": optimize proximity on all samples, but only apply the label-flip validity loss on negatives
        filter_mode = getattr(self, "cf_target_filter", "all")
        if filter_mode in ("neg_only", "validity_neg_only"):
            neg_mask = (y == 0)

            # loss on full batch
            l_1_all, l_2_all, l_3_all = call_loss(x, c, y, y_hat)

            # negative-only losses
            if neg_mask.any():
                x_neg = x[neg_mask]
                c_neg = c[neg_mask]
                y_neg = y[neg_mask]
                y_hat_neg = y_hat[neg_mask]
                _, l_2_neg, l_3_neg = call_loss(
                    x_neg, c_neg, y_neg, y_hat_neg,
                    _prox_override=(prox_c[neg_mask] if prox_c is not None else None),
                    _action_cost_override=(action_cost[neg_mask] if action_cost is not None else None),
                )
            else:
                device = y_hat.device
                l_2_neg = torch.tensor(0.0, device=device)
                l_3_neg = torch.tensor(0.0, device=device)

            if optimizer_idx == 0:
                result = self.predictor_step(l_1_all, l_3_all)
            else:
                if filter_mode == "neg_only":
                    # generator sees only negatives for both proximity and validity
                    result = self.explainer_step(l_2_neg, l_3_neg)
                    self._update_l2_tuning_ema(l_2_neg, l_3_neg)
                else:
                    # generator sees all samples for proximity, but only negatives for validity
                    result = self.explainer_step(l_2_all, l_3_neg)
                    self._update_l2_tuning_ema(l_2_all, l_3_neg)

            self._logging_loss(l_1_all, l_2_all, l_3_all, stage='train', on_step=False)
            return result

        if filter_mode == "flip_neg_stay_pos":
            l_1_all, l_2_all, _ = call_loss(x, c, y, y_hat)

            # recourse target: always positive
            y_prime = torch.ones_like(y_hat)

            _, _, l_3 = call_loss(x, c, y, y_hat, y_prime=y_prime)

            if optimizer_idx == 0:
                result = self.predictor_step(l_1_all, l_3)
            else:
                result = self.explainer_step(l_2_all, l_3)
                self._update_l2_tuning_ema(l_2_all, l_3)

            self._logging_loss(l_1_all, l_2_all, l_3, stage="train", on_step=False)
            return result

        # Default
        l_1, l_2, l_3 = call_loss(x, c, y, y_hat)

        if optimizer_idx == 0:
            result = self.predictor_step(l_1, l_3)
        else:
            result = self.explainer_step(l_2, l_3)
            self._update_l2_tuning_ema(l_2, l_3)

        self._logging_loss(l_1, l_2, l_3, stage='train', on_step=False)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # fwd
        y_hat, c = self(x, hard=True)

        # label for c
        if hasattr(self, "predict_proba") and callable(getattr(self, "predict_proba")):
            c_y = self.predict_proba(c)
        else:
            c_y, _ = self.model_forward(c)

        filter_mode = getattr(self, "cf_target_filter", "all")

        if filter_mode == "flip_neg_stay_pos":
            # flip negatives and don't flip positives
            y_prime = torch.ones_like(y_hat)
            l_1, l_2, l_3 = self._loss_functions(x, c, y, y_hat, y_prime=y_prime, is_val=True)

            cf_target = torch.ones_like(c_y)

        elif filter_mode == "validity_neg_only":
            # l1 on all, l2 on all, l3 only on negatives (y==0)
            l_1, l_2, _ = self._loss_functions(x, c, y, y_hat, is_val=True)

            neg_mask = (y == 0)
            if neg_mask.any():
                x_neg = x[neg_mask]
                c_neg = c[neg_mask]
                y_neg = y[neg_mask]
                y_hat_neg = y_hat[neg_mask]
                _, _, l_3 = self._loss_functions(x_neg, c_neg, y_neg, y_hat_neg, is_val=True)
            else:
                l_3 = torch.tensor(0.0, device=y_hat.device)

            cf_target = torch.ones_like(c_y)

        elif filter_mode == "neg_only":
            # l1 on all, (l2,l3) only on negatives (y==0)
            l_1, _, _ = self._loss_functions(x, c, y, y_hat, is_val=True)

            neg_mask = (y == 0)
            if neg_mask.any():
                x_neg = x[neg_mask]
                c_neg = c[neg_mask]
                y_neg = y[neg_mask]
                y_hat_neg = y_hat[neg_mask]
                _, l_2, l_3 = self._loss_functions(x_neg, c_neg, y_neg, y_hat_neg, is_val=True)
            else:
                l_2 = torch.tensor(0.0, device=y_hat.device)
                l_3 = torch.tensor(0.0, device=y_hat.device)

            cf_target = torch.ones_like(c_y)

        else:
            # flip predicted label
            l_1, l_2, l_3 = self._loss_functions(x, c, y, y_hat, is_val=True)
            cf_target = self.flip_prediction_scores(y_hat)

        loss = self.lambda_1 * l_1 + self.lambda_2 * l_2 + self.lambda_3 * l_3

        # logging val loss
        self._logging_loss(l_1, l_2, l_3, stage='val', on_step=False)

        # metrics
        metrics = {
            'val/val_loss': loss,
            'val/pred_accuracy': accuracy(y_hat, y.int()),
            'val/cf_proximity': proximity(x, c),
            'val/sensitivity': self.sensitivity(x, c, c_y),
            'val/cf_accuracy': accuracy(self.binarize_prediction_scores(c_y), cf_target.int()),
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        return loss

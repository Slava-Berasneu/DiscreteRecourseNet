__all__ = ['pl_logger', 'load_trained_model', 'coord_sparsity_and_manifold', 'ModelTrainer', 'CFGeneratorBase',
           'LocalCFGenerator', 'is_predictive_model', 'GlobalCFGenerator', 'Evaluator']

from .import_essentials import *
from .utils import *
from .training_module import BaseModule
from .model import BaselinePredictiveModel
from .base_interface import LocalExplainerBase, GlobalExplainerBase, ExplainerBase
from .evaluation import proximity
from .recourse_constraints import add_actionability_to_results
from sklearn.neighbors import NearestNeighbors
import json
import math

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
pl_logger = logging.getLogger("pytorch_lightning.core")

def load_trained_model(module: BaseModule, checkpoint_path: str, gpus: int = 0) -> BaseModule:
    """
    Load weights from a Lightning checkpoint into a constructed module.
    """
    checkpoint_path = str(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} is not found.")

    if gpus > 0 and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # Load weights into the provided module
    try:
        module.load_state_dict(state_dict)
    except RuntimeError:
        module.load_state_dict(state_dict, strict=False)

    module.to(device)
    module.eval()
    return module


def coord_sparsity_and_manifold(pred_model: BaseModule,
                                x: torch.Tensor,
                                cf: torch.Tensor,
                                target_y: Optional[torch.Tensor] = None,
                                ref_model: Optional[BaseModule] = None,
                                eps: float = 0.05,
                                n_neighbors: int = 1) -> tuple[float, float]:
    """
    Sparsity + manifold distance.

    Sparsity:
        1 - E[#(|cf - x| > eps) / d]

    Manifold distance:
        For each CF, compute L1 distance to the nearest neighbor among the
        training set restricted to label == target_y.
    """
    # Clamp to [0, 1]
    x_clamped = x.clamp(0.0, 1.0)
    cf_clamped = cf.clamp(0.0, 1.0)

    # Sparsity
    delta = (cf_clamped - x_clamped).abs()
    changed_mask = delta > eps
    changed_per_sample = changed_mask.sum(dim=1)
    d = x_clamped.size(-1)
    sparsity = 1.0 - (changed_per_sample / float(d)).mean().item()

    # Reference training set
    ref = ref_model if ref_model is not None else pred_model
    train_x, train_y = ref.train_dataset[:]
    train_x_clamped = train_x.clamp(0.0, 1.0)

    train_np = train_x_clamped.detach().cpu().numpy()
    train_y_np = train_y.detach().view(-1).cpu().numpy().astype(int)
    cf_np = cf_clamped.detach().cpu().numpy()

    target_np = target_y.detach().view(-1).cpu().numpy().astype(int)
    if cf_np.shape[0] != target_np.shape[0]:
        raise ValueError(f"target_y has length {target_np.shape[0]} but cf has {cf_np.shape[0]} samples")

    # Compute per-target-class NN distances
    dists_out = np.full((cf_np.shape[0],), np.nan, dtype=np.float32)
    for t in np.unique(target_np):
        cf_idx = np.where(target_np == t)[0]
        if cf_idx.size == 0:
            continue

        train_idx = np.where(train_y_np == t)[0]
        if train_idx.size == 0:
            continue

        k_eff = min(int(n_neighbors), int(train_idx.size))
        nn = NearestNeighbors(n_neighbors=k_eff, metric="manhattan")
        nn.fit(train_np[train_idx])
        dists, _ = nn.kneighbors(cf_np[cf_idx])
        dists_out[cf_idx] = dists[:, -1]  # kth-NN within target class

    manifold_dist = float(np.nanmean(dists_out)) if not np.all(np.isnan(dists_out)) else float('nan')
    return sparsity, manifold_dist


def _json_dumps(obj: Any) -> str:
    """Compact JSON for CSV cells"""
    def _default(o):
        try:
            if isinstance(o, (np.integer, np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.floating, np.float64, np.float32)):
                return float(o)
        except Exception:
            pass
        return str(o)
    return json.dumps(obj, ensure_ascii=False, default=_default)


def _decode_batch_to_feature_dicts(pred_model: BaseModule, X: torch.Tensor) -> List[Dict[str, Any]]:
    """Decode latent coordinate tensor into per-sample raw values"""
    cat_idx = pred_model.cat_normalizer.cat_idx
    cont_cols = list(getattr(pred_model, 'continous_cols', []))
    disc_cols = list(getattr(pred_model, 'discret_cols', []))

    X_cont_raw = pred_model.scaler.inverse_transform(X[:, :cat_idx])
    out: List[Dict[str, Any]] = []

    # categorical decoding
    cat_slices = getattr(pred_model.cat_normalizer, 'cat_slices', [])
    categories = getattr(pred_model.cat_normalizer, 'categories', [])

    for i in range(X.size(0)):
        d: Dict[str, Any] = {}
        # continuous
        for j, name in enumerate(cont_cols):
            v = float(X_cont_raw[i, j].item())
            if math.isnan(v):
                d[name] = None
            else:
                d[name] = v
        # categorical
        if categories:
            for k, name in enumerate(disc_cols):
                (s, e) = cat_slices[k]
                block = X[i, s:e]
                idx = int(block.argmax().item())
                cats = categories[k]
                try:
                    d[name] = cats[idx]
                except Exception:
                    d[name] = None
        out.append(d)
    return out


def _add_recourse_examples_to_results(
    *,
    results: Dict[str, Any],
    pred_model: BaseModule,
    configs: Dict[str, Any],
    x: torch.Tensor,
    cf: torch.Tensor,
    y_hat: Optional[torch.Tensor] = None,
    cf_y_hat: Optional[torch.Tensor] = None,
) -> None:
    """
    Attach a small sample of recourse suggestions to results.
    """
    try:
        n = int(x.size(0))
    except Exception:
        return
    if n <= 0:
        return

    n_examples = int(configs.get('recourse_examples_n', 10))
    seed = int(configs.get('recourse_examples_seed', 0))
    eps = float(configs.get('recourse_examples_eps', 1e-6))

    n_examples = max(0, min(n_examples, n))
    if n_examples == 0:
        results['recourse_examples'] = []
        return

    rng = np.random.default_rng(seed)
    idxs = rng.choice(n, size=n_examples, replace=False)
    idxs = [int(i) for i in idxs]

    x_dicts = _decode_batch_to_feature_dicts(pred_model, x[idxs])
    cf_dicts = _decode_batch_to_feature_dicts(pred_model, cf[idxs])

    examples: List[Dict[str, Any]] = []
    for row, i in enumerate(idxs):
        xd = x_dicts[row]
        cd = cf_dicts[row]
        changes = []
        for k in xd.keys():
            xv = xd.get(k, None)
            cv = cd.get(k, None)
            if xv is None and cv is None:
                continue
            if isinstance(xv, (int, float)) and isinstance(cv, (int, float)):
                if (not math.isnan(float(xv))) and (not math.isnan(float(cv))) and abs(float(cv) - float(xv)) > eps:
                    changes.append({'feature': k, 'from': float(xv), 'to': float(cv)})
            else:
                if xv != cv:
                    changes.append({'feature': k, 'from': xv, 'to': cv})

        ex = {
            'index': int(i),
            'x': xd,
            'cf': cd,
            'changes': changes,
        }
        if y_hat is not None:
            try:
                ex['y_hat'] = int(y_hat[i].item())
            except Exception:
                pass
        if cf_y_hat is not None:
            try:
                ex['cf_y_hat'] = int(cf_y_hat[i].item())
            except Exception:
                pass
        examples.append(ex)

    results['recourse_examples'] = examples


class ModelTrainer(object):
    def __init__(self,
                 model: BaseModule,
                 t_configs: Dict[str, Any],
                 callbacks: Optional[List[Callback]] = None,
                 description: Optional[str] = None,
                 debug: Optional[bool] = False,
                 logger: Optional[Union[LightningLoggerBase, bool]] = None,
                 logger_name: str = "debug"):

        if logger is None:
            logger = pl_loggers.TensorBoardLogger(
                save_dir="log/",
                name=logger_name,
                log_graph=False,
                default_hp_metric=False,
            )


        # model checkpoint
        self.checkpoint_callback = ModelCheckpoint(
            monitor='val/val_loss', save_top_k=3, mode='min'
        )

        # define callbacks
        if callbacks is None:
            callbacks = [self.checkpoint_callback]
        elif self._has_no_model_checkpoint(callbacks):
            callbacks += [self.checkpoint_callback]

        self.trainer = pl.Trainer(logger=logger, callbacks=callbacks, **t_configs)

        self.model = model

    def _has_no_model_checkpoint(self, callbacks: List[Callback]) -> bool:
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                return False
        return True

    def fit(self, is_parallel=False):
        if is_parallel:
            logging.warning(
                f"parallel version has not been implemented\nUsing the single process training...")
        self.trainer.fit(self.model)

        return self.model

    def save_best_model(self, dir_path: Path):
        if not dir_path.is_dir():
            raise ValueError(f"'{dir_path}' is not a directory")

        best_model_path = Path(self.checkpoint_callback.best_model_path)
        if not best_model_path.is_file():
            raise FileNotFoundError(f"Best model checkpoint not found at {best_model_path}")

        dest_path = dir_path / best_model_path.name
        shutil.copy(best_model_path, dest_path)

        # create a per-model checkpoint
        alias_path = dir_path / f"best_{self.model.__class__.__name__}.ckpt"
        shutil.copy(dest_path, alias_path)

        return dest_path

    def load_trained_model(self, checkpoint_path: str, gpus: int = 0) -> BaseModule:
        self.model = load_trained_model(
            self.model, checkpoint_path=checkpoint_path, gpus=gpus)
        return self.model


class CFGeneratorBase(ABC):
    results = {
        "x": None,
        "cf": None,
        "y": None,
        "y_hat": None,
        "cf_y": None,
        "cf_y_hat": None,
        "sensitivity": None,
        "total_time": None,
        "avg_time": None,
        "cf_algo": None,
        "cat_idx": None,
        "manifold_dist": None,
        "sparsity": None,
    }

    def __init__(self, cf_algo: ExplainerBase,
                 pred_model: BaselinePredictiveModel,
                 configs: Dict[str, Any] = {}, ref_model:
                Optional[BaselinePredictiveModel] = None):
        self.configs = configs
        self.pred_model = pred_model
        self.ref_model = ref_model if ref_model is not None else pred_model
        self.pred_model.freeze()

        self.cf_algo = cf_algo
        self.results.update({"cf_algo": type(cf_algo).__name__})
        self.dataset = pred_model.test_dataset
        self.sensitivity = pred_model.sensitivity

        try:
            self.results.update({'cat_idx': int(pred_model.cat_normalizer.cat_idx)})
        except Exception:
            self.results.update({'cat_idx': None})

    def _resolve_dataset_and_size(
        self,
        dataset: Optional[TensorDataset],
        test_size: Optional[int],
        debug: bool,
    ) -> tuple[TensorDataset, int]:
        if dataset is None:
            dataset = self.pred_model.test_dataset
        if test_size is None:
            size = len(dataset) if not debug else min(3, len(dataset))
        else:
            size = min(int(test_size), len(dataset))
        return dataset, size

    def _target_cf_labels(self, y_hat: torch.Tensor) -> torch.Tensor:
        if self.configs.get("cf_target_filter") == "flip_neg_stay_pos":
            return torch.ones_like(y_hat)
        return flip_binary(y_hat)

    def _finalize_results(
        self,
        *,
        x: torch.Tensor,
        cf: torch.Tensor,
        y: torch.Tensor,
        total_time: float,
        avg_time: float,
    ) -> Dict[str, Any]:
        y_hat = self.pred_model.predict(x)
        cf_y_hat = self.pred_model.predict(cf)

        sensitivity = self.pred_model.sensitivity
        sensitivity.reset()

        cf_y = self._target_cf_labels(y_hat)

        self.results.update({
            'x': x,
            'cf': cf,
            'y': y,
            'y_hat': y_hat,
            'cf_y': cf_y,
            'cf_y_hat': cf_y_hat,
            'sensitivity': sensitivity(x, cf, cf_y).item(),
            'total_time': total_time,
            'avg_time': avg_time,
        })

        eps = self.configs.get('sparsity_eps', 0.05)
        k = self.configs.get('manifold_k', 1)
        neg_mask = (y_hat == 0).reshape(-1)

        if neg_mask.sum() > 0:
            sparsity_val, man_dist = coord_sparsity_and_manifold(
                self.pred_model,
                x[neg_mask],
                cf[neg_mask],
                target_y=cf_y[neg_mask],
                ref_model=self.ref_model,
                eps=eps,
                n_neighbors=k,
            )
        else:
            sparsity_val, man_dist = float('nan'), float('nan')

        self.results.update({
            'sparsity': sparsity_val,
            'manifold_dist': man_dist,
        })

        add_actionability_to_results(
            results=self.results,
            pred_model=self.pred_model,
            configs=self.configs,
            x=x,
            cf=cf,
            y_hat=y_hat,
        )

        _add_recourse_examples_to_results(
            results=self.results,
            pred_model=self.pred_model,
            configs=self.configs,
            x=x,
            cf=cf,
            y_hat=y_hat,
            cf_y_hat=cf_y_hat,
        )

        return self.results

    def generate(self, dataset: Optional[TensorDataset]=None, test_size: Optional[int] = None, debug: bool = False):
        raise NotImplementedError


class LocalCFGenerator(CFGeneratorBase):
    def __init__(self, cf_algo: LocalExplainerBase,
        pred_model: BaselinePredictiveModel,
        configs: Dict[str, Any] = {},
        ref_model: Optional[BaselinePredictiveModel] = None):
        super().__init__(cf_algo, pred_model, configs, ref_model=ref_model)
        # define cf_algo
        if not issubclass(type(cf_algo), LocalExplainerBase):
            raise ValueError(f"cf_algo should be an instance of `{LocalExplainerBase}`, but got `{type(cf_algo)}`. ")
        CFExplainer = type(cf_algo)
        pred_fn = pred_model.forward
        cat_normalizer = pred_model.cat_normalizer
        self.cf_algo = CFExplainer(pred_fn, cat_normalizer, configs)

        self.is_parallel = configs['is_parallel'] if 'is_parallel' in configs else True

    def gen_step(self, x):
        x = x.reshape(1, -1)
        cf = self.cf_algo.generate_cf(x)
        return x, cf

    def iterative_generate(self, size: int, dataset: TensorDataset):
        result = []
        start_time = time.time()
        for ix, (x, y) in enumerate(tqdm(dataset)):
            if ix < size:
                x, cf = self.gen_step(x)
                result.append((x, cf))
        total_time = time.time() - start_time
        avg_time = total_time / size
        return result, {'total_time': total_time, 'avg_time': avg_time}

    def __unpack_x_cf(self, result: List[torch.Tensor]):
        X = torch.rand((len(result), result[0][0].size(-1)))
        cf_algo = X.clone()

        for ix, (x, cf) in enumerate(result):
            X[ix, :] = x
            cf_algo[ix, :] = cf
        return X, cf_algo

    def generate(self, dataset: Optional[TensorDataset] = None, test_size: Optional[int] = None, debug: bool = False):
        dataset, size = self._resolve_dataset_and_size(dataset, test_size, debug)

        result = []

        if self.is_parallel and not debug:
            print(f"generating {size} cfs in parallel...")
            result = Parallel(n_jobs=12, max_nbytes=None, verbose=False, backend="threading")(
                delayed(self.gen_step) (x=x)
                for ix, (x, y) in enumerate(tqdm(dataset)) if ix < size
            )
            print(f"evaluating speed by generating 50 cfs...")
            _, time = self.iterative_generate(50, dataset)
        else:
            print(f"generating {size} cfs...")
            result, time = self.iterative_generate(size, dataset)

        x, cf = self.__unpack_x_cf(result)
        _, y = dataset[:]
        y = y[:size]
        return self._finalize_results(
            x=x,
            cf=cf,
            y=y,
            total_time=float(time['total_time']),
            avg_time=float(time['avg_time']),
        )


def is_predictive_model(model: BaseModule):
    return callable(getattr(model, "predict", None))


class GlobalCFGenerator(CFGeneratorBase):
    def __init__(self, cf_algo: GlobalExplainerBase,
            pred_model: Optional[BaselinePredictiveModel] = None, configs: Dict[str, Any] = {},
                 ref_model: Optional[BaselinePredictiveModel] = None) -> None:
        if not issubclass(type(cf_algo), GlobalExplainerBase):
            raise ValueError(f"cf_algo should be an instance of `{GlobalCFGenerator}`, but got `{type(cf_algo)}`")
        if not is_predictive_model(cf_algo) and pred_model is None:
            raise ValueError(f"pred_model should be passed when cf_algo is {type(cf_algo)}.")
        if is_predictive_model(cf_algo):
            pred_model = cf_algo
        super().__init__(cf_algo, pred_model, configs, ref_model=ref_model)

    def generate(self, dataset: Optional[TensorDataset]=None, test_size: Optional[int] = None, debug: bool = False):
        dataset, size = self._resolve_dataset_and_size(dataset, test_size, debug)
        x, y = dataset[:]
        x = x[:size]
        y = y[:size]

        print(f"generating {size} cfs...")
        cf = self.cf_algo.generate_cf(x)

        print(f"evaluating speed...")
        start_time = time.time()
        for i, (sample, _) in enumerate(dataset):
            if i < size:
                self.cf_algo.generate_cf(sample.reshape(1, -1))
        total_time = time.time() - start_time
        avg_time = total_time / size
        return self._finalize_results(
            x=x,
            cf=cf,
            y=y,
            total_time=total_time,
            avg_time=avg_time,
        )

# Cell
class Evaluator(object):
    def __init__(self, configs: Dict[str, Any]={}):
        self.is_logging: bool = configs.get('is_logging', True)

    def eval(self, results: Dict[str, Any], dir_path: Path):
        if not dir_path.exists():
            raise ValueError(f"{dir_path} does not exist.")
        csv_path = dir_path / Path('metrics.csv')

        metrics = [
            'cat_proximity',
            'cont_proximity',
            #'validity',
            'sensitivity',
            'time',
            'pred_accuracy',
            'proximity',
            'sparsity',
            'manifold_dist',
            'validity_recourse', # fraction of y_hat==0 that reach class 1
            # actionability
            'actionability_rate',
            'monotonicity_violation_rate',
            'immutability_violation_rate',
            'group_rule_violation_rate',
            'avg_num_actionability_violations',
            'avg_num_actionability_changes',
            'valid_change_rate',
            #'validity_goal_pos',  # y_hat==0 reach 1, y_hat==1 stay 1
        ]
        # ['diffs', 'total_num']

        if csv_path.exists():
            r = pd.read_csv(csv_path, index_col=0).to_dict()
            for metric in metrics:
                if metric not in r.keys():
                    r[metric] = dict()
        else:
            r = {metric:{} for metric in metrics}

        x, cf, y, y_hat, cf_y, cf_y_hat = (
            results['x'],
            results['cf'],
            results['y'],
            results['y_hat'],
            results['cf_y'],
            results['cf_y_hat'],
        )
        cat_idx, cf_name = results['cat_idx'], results['cf_algo']

        # --- Proximity ---
        neg_mask = (y_hat == 0).reshape(-1)

        if neg_mask.sum() > 0:
            x_prox = x[neg_mask]
            cf_prox = cf[neg_mask]
        else:
            x_prox = x
            cf_prox = cf
            print(f"[WARN] No negative predictions found for {cf_name}. Proximity computed on entire set.")

        cont_prox = proximity(x_prox[:, :cat_idx], cf_prox[:, :cat_idx]).item()
        cat_prox = proximity(x_prox[:, cat_idx:], cf_prox[:, cat_idx:]).item()
        r['cont_proximity'][cf_name] = cont_prox
        r['cat_proximity'][cf_name] = cat_prox

        # divide by number of features
        n_features = x.size(-1)
        r['proximity'][cf_name] = (cont_prox + cat_prox) / float(n_features)

        # other metrics
        #r['validity'][cf_name] = accuracy(cf_y.int(), cf_y_hat.int()).item()
        r['sensitivity'][cf_name] = results['sensitivity']
        r['time'][cf_name] = results['avg_time']
        r['pred_accuracy'][cf_name] = accuracy(y.int(), y_hat.int()).item()

        # sparsity + manifold distance
        r['sparsity'][cf_name] = float(results['sparsity'])
        r['manifold_dist'][cf_name] = float(results['manifold_dist'])

        # do CFs reach the positive class for those predicted as negative
        neg_mask = (y_hat == 0)
        if neg_mask.any():
            target_pos = torch.ones_like(cf_y_hat[neg_mask])
            r['validity_recourse'][cf_name] = accuracy(target_pos.int(), cf_y_hat[neg_mask].int()).item()
        else:
            r['validity_recourse'][cf_name] = float('nan')

        # Actionability metrics
        for k in (
            'actionability_rate',
            'monotonicity_violation_rate',
            'immutability_violation_rate',
            'group_rule_violation_rate',
            'avg_num_actionability_violations',
            'avg_num_actionability_changes',
            'valid_change_rate',
        ):
            r[k][cf_name] = float(results.get(k, float('nan')))

        #target_all_pos = torch.ones_like(cf_y_hat)
        #r['validity_goal_pos'][cf_name] = accuracy(target_all_pos.int(), cf_y_hat.int()).item()

        final_result_df = pd.DataFrame.from_dict(r)
        print(tabulate(final_result_df.astype("float16"), headers = 'keys', tablefmt = 'pretty'))
        if self.is_logging:
            final_result_df.to_csv(csv_path)
            torch.save(results, dir_path / f"{cf_name}_results.pt")

            # Recourse examples (same examples as handled by different models)
            try:
                examples = results.get('recourse_examples', None)
                if isinstance(examples, list) and len(examples) > 0:
                    ex_path = dir_path / "recourse_examples.csv"
                    write_header = not ex_path.exists()
                    import csv as _csv
                    with open(ex_path, "a", newline="", encoding="utf-8") as f:
                        w = _csv.writer(f)
                        if write_header:
                            w.writerow(["model", "index", "y_hat", "cf_y_hat", "changes_json", "x_json", "cf_json"])
                        for ex in examples:
                            w.writerow([
                                cf_name,
                                ex.get("index", None),
                                ex.get("y_hat", None),
                                ex.get("cf_y_hat", None),
                                _json_dumps(ex.get("changes", [])),
                                _json_dumps(ex.get("x", {})),
                                _json_dumps(ex.get("cf", {})),
                            ])
            except Exception as e:
                print(f"[WARN] Could not write recourse examples for {cf_name}: {e}")

            # Per-group breakdown
            gbd = results.get('actionability_group_breakdown', None)
            if isinstance(gbd, dict) and len(gbd) > 0:
                try:
                    gdf = pd.DataFrame.from_dict(gbd, orient='index')
                    sort_cols = [c for c in [
                        'rule_violation_rate',
                        'monotonicity_violation_rate',
                        'immutability_violation_rate',
                        'change_rate',
                    ] if c in gdf.columns]
                    if sort_cols:
                        gdf = gdf.sort_values(by=sort_cols, ascending=False)
                    gdf.to_csv(dir_path / f"{cf_name}_actionability_groups.csv")
                except Exception as e:
                    print(f"[WARN] Could not write group breakdown CSV for {cf_name}: {e}")
            print("Results have been saved!")
        return final_result_df



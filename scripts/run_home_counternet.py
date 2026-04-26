from pathlib import Path
import argparse
import sys
import itertools
import shutil
import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from pytorch_lightning import seed_everything

from counternet.dataset import load_configs
from counternet.model import (
    CounterNetModel,
    CounterNetProjectionModel,
    BaselinePredictiveModel,
    DiscreteRecourseNetModel,
)
from counternet.pipeline import (
    GlobalCFGenerator,
    LocalCFGenerator,
    Evaluator,
    ModelTrainer,
    load_trained_model,
)
from counternet.cf_explainer import VanillaCF
from counternet.training_module import CFNetTrainingModule


# Shared-mode model selection order.
MODELS_TO_RUN: List[str] = [
    "discrete_recoursenet",
    "counternet_projection",
    "counternet",
    #"vanillacf"
]

# Comparison-mode per-model settings
# These are ignored in shared mode and all models use the same config.
MODEL_RUN_SETTINGS: Dict[str, Dict[str, Any]] = {
    "discrete_recoursenet": {
         "ablation": "cfgen_flip_neg_stay_pos",
         "overrides": {
            "lambda_1":  1.0,
            "lambda_2":  0.02,
            "gumbel_tau_mask":  0.5,
            "gumbel_tau_choice":  0.75,
            "action_cost_base":  0.005,
            "lambda_3":  1.0,
         }

    },
    "counternet_projection": {
        "ablation": "cfgen_flip_neg_stay_pos"
        # "ablation": "cfgen_flip_neg_stay_pos",
    },
    "counternet": {
        "ablation": "cfgen_flip_neg_stay_pos"
        # "ablation": "cfgen_flip_neg_stay_pos",
    },
    #"vanillacf": {}
}

MODEL_REGISTRY = {
    "counternet": CounterNetModel,
    "counternet_projection": CounterNetProjectionModel,
    "discrete_recoursenet": DiscreteRecourseNetModel,
    "vanillacf": VanillaCF,
}


@dataclass(frozen=True)
class ModelRunSpec:
    name: str
    label: str
    ablation: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "ablation": self.ablation,
            "overrides": dict(self.overrides),
        }


def resolve_model_class(model_name: str) -> type:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Options: {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]


def resolve_ablation_tag(m_config: dict, ablation: Optional[str]) -> str:
    if ablation:
        return ablation
    return m_config.get("ablation_tag") or "default"


def resolve_dataset_config_paths(
    dataset_config: Path,
    dataset_configs: Optional[List[Path]] = None,
    datasets: Optional[List[str]] = None,
) -> List[Path]:
    """Resolve which dataset config JSON files to run."""
    if dataset_configs:
        return [Path(p) for p in dataset_configs]

    if datasets:
        resolved: List[Path] = []
        for name in datasets:
            p_main = PROJECT_ROOT / "assets" / "configs" / f"{name}.json"
            p_extra = PROJECT_ROOT / "assets" / "configs" / "extra" / f"{name}.json"
            resolved.append(p_main if p_main.exists() else p_extra if p_extra.exists() else p_main)
        return resolved

    return [Path(dataset_config)]


def apply_ablation_to_config(
    m_config: dict,
    ablation: Optional[str],
    *,
    set_tag: bool = True,
) -> dict:
    """Model ablation options."""
    if ablation is None:
        return m_config

    m_config = dict(m_config)

    if ablation == "cfgen_neg_only":
        m_config["cf_target_filter"] = "neg_only"
    elif ablation == "cfgen_validity_neg_only":
        m_config["cf_target_filter"] = "validity_neg_only"
    elif ablation == "cfgen_flip_neg_stay_pos":
        m_config["cf_target_filter"] = "flip_neg_stay_pos"
    else:
        raise ValueError(f"Unknown ablation mode: {ablation}")

    if set_tag:
        m_config["ablation_tag"] = ablation
    return m_config


def _parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return [float(x) for x in s.split(",") if str(x).strip()]


def _fmt(v: float) -> str:
    s = f"{float(v):g}"
    return s.replace("-", "m").replace(".", "p")


def _clean_run_dir(dir_path: Path, *, keep_ckpts: bool) -> None:
    if not dir_path.exists():
        return
    if not keep_ckpts:
        shutil.rmtree(dir_path)
        return

    # Keep checkpoints, drop other artifacts
    for pat in ("metrics.csv", "run_manifest.json", "recourse_examples.csv", "*_results.pt", "*_actionability_groups.csv"):
        for p in dir_path.glob(pat):
            try:
                p.unlink()
            except Exception:
                pass


def _resolve_project_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _absolutize_model_config_paths(m_config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(m_config)
    if "data_dir" in resolved and resolved["data_dir"] is not None:
        resolved["data_dir"] = str(_resolve_project_path(resolved["data_dir"]))
    resolved.setdefault(
        "action_groups_path",
        str(PROJECT_ROOT / "assets" / "actions" / "action_groups.json"),
    )
    return resolved


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    slug = slug.strip("._")
    return slug or "model"


def _normalize_model_run_spec(model_name: str, settings: Optional[Dict[str, Any]]) -> ModelRunSpec:
    raw = dict(settings or {})
    label = str(raw.pop("label", model_name))
    ablation = raw.pop("ablation", None)
    overrides = dict(raw.pop("overrides", {}) or {})
    if raw:
        unknown = ", ".join(sorted(raw.keys()))
        raise ValueError(
            f"Unsupported MODEL_RUN_SETTINGS keys for '{model_name}': {unknown}. "
            f"Supported keys: label, ablation, overrides."
        )
    return ModelRunSpec(name=model_name, label=label, ablation=ablation, overrides=overrides)


def resolve_model_run_specs(*, run_mode: str) -> List[ModelRunSpec]:
    specs: List[ModelRunSpec] = []
    labels_seen: set[str] = set()

    for model_name in MODELS_TO_RUN:
        resolve_model_class(model_name)
        settings = MODEL_RUN_SETTINGS.get(model_name) if run_mode == "comparison" else None
        spec = _normalize_model_run_spec(model_name, settings)
        if spec.label in labels_seen:
            raise ValueError(
                f"Duplicate model label '{spec.label}'. Labels must be unique within one run."
            )
        labels_seen.add(spec.label)
        specs.append(spec)

    return specs


def build_model_config(
    *,
    base_config: Dict[str, Any],
    spec: ModelRunSpec,
    default_ablation: Optional[str],
    run_mode: str,
    shared_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    effective_ablation = spec.ablation if spec.ablation is not None else default_ablation

    cfg = dict(base_config)
    cfg = apply_ablation_to_config(cfg, effective_ablation, set_tag=(run_mode == "shared"))

    if shared_overrides:
        cfg.update(shared_overrides)
    if spec.overrides:
        cfg.update(spec.overrides)

    cfg = _absolutize_model_config_paths(cfg)
    cfg["_runner_model_name"] = spec.name
    cfg["_runner_model_label"] = spec.label
    cfg["_runner_effective_ablation"] = effective_ablation
    return cfg


def resolve_run_tag(
    *,
    base_config: Dict[str, Any],
    run_mode: str,
    default_ablation: Optional[str],
    run_tag: Optional[str],
    extra_tag: Optional[str] = None,
) -> str:
    if run_tag:
        tag = run_tag
    elif run_mode == "shared":
        tag = resolve_ablation_tag(base_config, default_ablation)
    else:
        tag = "comparison"

    if extra_tag:
        tag = f"{tag}__{extra_tag}"
    return tag


def resolve_run_dir(
    run_dir_arg: Optional[str],
    *,
    dataset_name: str,
    results_root: Path,
    seed: int,
    run_tag: str,
) -> Path:
    """<results_root>/<run_tag>/<dataset_name>/seed-<seed>/"""
    if run_dir_arg:
        p = Path(run_dir_arg)
        if not p.exists():
            raise FileNotFoundError(f"Run directory does not exist: {p}")
        if not p.is_dir():
            raise ValueError(f"--ckpt must point to a *run directory* (not a file). Got: {p}")
        return p

    return results_root / run_tag / dataset_name / f"seed-{seed}"


def checkpoint_dir_for_spec(run_dir: Path, spec: ModelRunSpec, *, run_mode: str) -> Path:
    if run_mode == "comparison":
        return run_dir / "models" / _slugify_name(spec.label)
    return run_dir


def find_checkpoint_for_spec(
    *,
    run_dir: Path,
    spec: ModelRunSpec,
    model_cls: type,
) -> Path:
    candidate_dirs = [
        run_dir / "models" / _slugify_name(spec.label),
        run_dir,
    ]

    candidate_names = [
        "best.ckpt",
        f"best_{model_cls.__name__}.ckpt",
    ]

    for base_dir in candidate_dirs:
        for name in candidate_names:
            checkpoint = base_dir / name
            if checkpoint.is_file():
                return checkpoint

    raise FileNotFoundError(
        f"No checkpoint found for label='{spec.label}' ({model_cls.__name__}) in {run_dir}. "
        f"Try running with --retrain first."
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def _needs_baseline_predictive_model(specs: List[ModelRunSpec]) -> bool:
    for spec in specs:
        model_cls = resolve_model_class(spec.name)
        if not issubclass(model_cls, CFNetTrainingModule):
            return True
    return False


def _train_baseline_predictive_model(
    *,
    base_config: Dict[str, Any],
    t_config: Dict[str, Any],
) -> BaselinePredictiveModel:
    print("[PRED] Training baseline predictive model for local explanations...")
    pred_config = _absolutize_model_config_paths(dict(base_config))
    pred_model = BaselinePredictiveModel(pred_config)
    pred_trainer = ModelTrainer(pred_model, t_config, logger_name="pred_model")
    return pred_trainer.fit()


def _prepare_spec_artifact_dir(
    *,
    run_dir: Path,
    spec: ModelRunSpec,
    run_mode: str,
    m_config: Dict[str, Any],
) -> Path:
    model_dir = checkpoint_dir_for_spec(run_dir, spec, run_mode=run_mode)
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_json(model_dir / "resolved_config.json", m_config)
    return model_dir


def _build_local_vanillacf_generator(
    *,
    pred_model: Optional[BaselinePredictiveModel],
    m_config: Dict[str, Any],
) -> LocalCFGenerator:
    if pred_model is None:
        raise RuntimeError("VanillaCF requires a baseline predictive model.")

    local_cfg = dict(m_config)
    local_cfg.pop("cf_target_filter", None)
    local_cfg["lr"] = 0.05
    local_cfg["max_iter"] = 1000
    return LocalCFGenerator(
        VanillaCF(pred_model.predict),
        pred_model,
        configs=local_cfg,
        ref_model=pred_model,
    )


def _evaluate_spec_generator(
    *,
    spec: ModelRunSpec,
    cf_generator: Union[LocalCFGenerator, GlobalCFGenerator],
    evaluator: Evaluator,
    run_dir: Path,
    debug: bool,
) -> None:
    results = cf_generator.generate(debug=debug)
    results["cf_algo"] = spec.label
    evaluator.eval(results, run_dir)


def _train_and_generate_for_spec(
    *,
    spec: ModelRunSpec,
    m_config: Dict[str, Any],
    t_config: Dict[str, Any],
    run_dir: Path,
    run_mode: str,
    debug: bool,
    pred_model: Optional[BaselinePredictiveModel],
    evaluator: Evaluator,
) -> Optional[Path]:
    model_cls = resolve_model_class(spec.name)
    ckpt_path: Optional[Path] = None
    model_dir = _prepare_spec_artifact_dir(
        run_dir=run_dir,
        spec=spec,
        run_mode=run_mode,
        m_config=m_config,
    )

    if issubclass(model_cls, CFNetTrainingModule):
        model = model_cls(m_config)
        logger_name = f"{_slugify_name(spec.label).lower()}/{m_config['dataset_name']}"
        trainer = ModelTrainer(model, t_config, logger_name=logger_name)
        trainer.fit()

        best_model_path = trainer.save_best_model(model_dir)
        ckpt_path = model_dir / "best.ckpt"
        shutil.copy(best_model_path, ckpt_path)

        model = trainer.load_trained_model(checkpoint_path=str(best_model_path), gpus=0)
        ref_model = pred_model if pred_model is not None else model
        cf_generator = GlobalCFGenerator(model, configs=m_config, ref_model=ref_model)

    else:
        if model_cls is not VanillaCF:
            raise NotImplementedError(
                f"Runner does not support non-CFNet explainer {model_cls.__name__}"
            )
        cf_generator = _build_local_vanillacf_generator(
            pred_model=pred_model,
            m_config=m_config,
        )

    _evaluate_spec_generator(
        spec=spec,
        cf_generator=cf_generator,
        evaluator=evaluator,
        run_dir=run_dir,
        debug=debug,
    )
    return ckpt_path


def _load_and_generate_for_spec(
    *,
    spec: ModelRunSpec,
    m_config: Dict[str, Any],
    run_dir: Path,
    run_mode: str,
    debug: bool,
    pred_model: Optional[BaselinePredictiveModel],
    evaluator: Evaluator,
) -> None:
    model_cls = resolve_model_class(spec.name)
    _prepare_spec_artifact_dir(
        run_dir=run_dir,
        spec=spec,
        run_mode=run_mode,
        m_config=m_config,
    )

    if issubclass(model_cls, CFNetTrainingModule):
        ckpt_path = find_checkpoint_for_spec(run_dir=run_dir, spec=spec, model_cls=model_cls)
        print(f"[EVAL] Loading {spec.label} checkpoint from: {ckpt_path}")
        model = model_cls(m_config)
        model.prepare_data()
        model = load_trained_model(model, checkpoint_path=str(ckpt_path), gpus=0)
        ref_model = pred_model if pred_model is not None else model
        cf_generator = GlobalCFGenerator(model, configs=m_config, ref_model=ref_model)
    else:
        if model_cls is not VanillaCF:
            raise NotImplementedError(
                f"Runner does not support non-CFNet explainer {model_cls.__name__}"
            )
        cf_generator = _build_local_vanillacf_generator(
            pred_model=pred_model,
            m_config=m_config,
        )

    _evaluate_spec_generator(
        spec=spec,
        cf_generator=cf_generator,
        evaluator=evaluator,
        run_dir=run_dir,
        debug=debug,
    )


def _write_run_manifest(
    *,
    run_dir: Path,
    dataset_config_path: Path,
    trainer_config_path: Path,
    dataset_name: str,
    seed: int,
    run_mode: str,
    run_tag: str,
    default_ablation: Optional[str],
    specs: List[ModelRunSpec],
    base_config: Dict[str, Any],
    shared_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    models_payload = []
    for spec in specs:
        resolved_cfg = build_model_config(
            base_config=base_config,
            spec=spec,
            default_ablation=default_ablation,
            run_mode=run_mode,
            shared_overrides=shared_overrides,
        )
        models_payload.append({
            "spec": spec.as_dict(),
            "checkpoint_dir": str(checkpoint_dir_for_spec(run_dir, spec, run_mode=run_mode)),
            "resolved_config_path": str(checkpoint_dir_for_spec(run_dir, spec, run_mode=run_mode) / "resolved_config.json"),
            "effective_ablation": resolved_cfg.get("_runner_effective_ablation"),
        })

    _write_json(run_dir / "run_manifest.json", {
        "dataset_config_path": str(dataset_config_path),
        "trainer_config_path": str(trainer_config_path),
        "dataset_name": dataset_name,
        "seed": int(seed),
        "run_mode": run_mode,
        "run_tag": run_tag,
        "default_ablation": default_ablation,
        "shared_overrides": dict(shared_overrides or {}),
        "models": models_payload,
    })


def train_full_experiment(
    m_config_path: Path,
    t_config_path: Path,
    results_root: Path,
    seed: int,
    debug: bool,
    ablation: Optional[str],
    *,
    extra_tag: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    run_mode: str = "shared",
    run_tag: Optional[str] = None,
) -> None:
    # Clean existing seed directory. Then train + generate CFs + eval
    seed_everything(seed, workers=True)

    base_config = load_configs(m_config_path)
    t_config = load_configs(t_config_path)

    specs = resolve_model_run_specs(run_mode=run_mode)
    effective_run_tag = resolve_run_tag(
        base_config=base_config,
        run_mode=run_mode,
        default_ablation=ablation,
        run_tag=run_tag,
        extra_tag=extra_tag,
    )

    run_dir = resolve_run_dir(
        None,
        dataset_name=base_config["dataset_name"],
        results_root=results_root,
        seed=seed,
        run_tag=effective_run_tag,
    )
    _clean_run_dir(run_dir, keep_ckpts=False)
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_run_manifest(
        run_dir=run_dir,
        dataset_config_path=m_config_path,
        trainer_config_path=t_config_path,
        dataset_name=base_config["dataset_name"],
        seed=seed,
        run_mode=run_mode,
        run_tag=effective_run_tag,
        default_ablation=ablation,
        specs=specs,
        base_config=base_config,
        shared_overrides=overrides if run_mode == "shared" else None,
    )

    print(
        f"[TRAIN] Running {run_mode} experiment on dataset='{base_config['dataset_name']}', "
        f"seed={seed}, run_tag={effective_run_tag}"
    )

    pred_model = None
    if _needs_baseline_predictive_model(specs):
        pred_model = _train_baseline_predictive_model(base_config=base_config, t_config=t_config)

    evaluator = Evaluator(configs={"is_logging": True})

    for spec in specs:
        model_config = build_model_config(
            base_config=base_config,
            spec=spec,
            default_ablation=ablation,
            run_mode=run_mode,
            shared_overrides=overrides if run_mode == "shared" else None,
        )
        effective_ablation = model_config.get("_runner_effective_ablation")
        print(
            f"[TRAIN] Model='{spec.label}' ({spec.name}), "
            f"ablation={effective_ablation or 'default'}, "
            f"overrides={spec.overrides if run_mode == 'comparison' else (overrides or {})}"
        )
        ckpt_path = _train_and_generate_for_spec(
            spec=spec,
            m_config=model_config,
            t_config=t_config,
            run_dir=run_dir,
            run_mode=run_mode,
            debug=debug,
            pred_model=pred_model,
            evaluator=evaluator,
        )
        if ckpt_path is not None:
            print(f"[TRAIN] Saved checkpoint for '{spec.label}' to {ckpt_path}")


def eval_from_checkpoint(
    m_config_path: Path,
    t_config_path: Path,
    results_root: Path,
    seed: int,
    run_dir_arg: Optional[str],
    debug: bool,
    ablation: Optional[str],
    *,
    run_mode: str = "shared",
    run_tag: Optional[str] = None,
) -> None:
    # Evaluate a previously trained run: regenerate CFs + compute metrics
    seed_everything(seed, workers=True)

    base_config = load_configs(m_config_path)
    t_config = load_configs(t_config_path)

    specs = resolve_model_run_specs(run_mode=run_mode)
    effective_run_tag = resolve_run_tag(
        base_config=base_config,
        run_mode=run_mode,
        default_ablation=ablation,
        run_tag=run_tag,
    )

    run_dir = resolve_run_dir(
        run_dir_arg,
        dataset_name=base_config["dataset_name"],
        results_root=results_root,
        seed=seed,
        run_tag=effective_run_tag,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _clean_run_dir(run_dir, keep_ckpts=True)

    _write_run_manifest(
        run_dir=run_dir,
        dataset_config_path=m_config_path,
        trainer_config_path=t_config_path,
        dataset_name=base_config["dataset_name"],
        seed=seed,
        run_mode=run_mode,
        run_tag=effective_run_tag,
        default_ablation=ablation,
        specs=specs,
        base_config=base_config,
        shared_overrides=None,
    )

    pred_model = None
    if _needs_baseline_predictive_model(specs):
        pred_model = _train_baseline_predictive_model(base_config=base_config, t_config=t_config)

    evaluator = Evaluator(configs={"is_logging": True})

    for spec in specs:
        model_config = build_model_config(
            base_config=base_config,
            spec=spec,
            default_ablation=ablation,
            run_mode=run_mode,
            shared_overrides=None,
        )
        effective_ablation = model_config.get("_runner_effective_ablation")
        print(
            f"[EVAL] Model='{spec.label}' ({spec.name}), "
            f"ablation={effective_ablation or 'default'}"
        )
        _load_and_generate_for_spec(
            spec=spec,
            m_config=model_config,
            run_dir=run_dir,
            run_mode=run_mode,
            debug=debug,
            pred_model=pred_model,
            evaluator=evaluator,
        )

    print(f"[DONE] Evaluation finished. See {run_dir / 'metrics.csv'}")


def run_grid_search(
    m_config_path: Path,
    t_config_path: Path,
    results_root: Path,
    seed: int,
    debug: bool,
    ablation: Optional[str],
    grid: Dict[str, List[float]],
    limit: Optional[int] = None,
    *,
    run_tag: Optional[str] = None,
) -> None:
    base = load_configs(m_config_path)
    base_tag = resolve_run_tag(
        base_config=base,
        run_mode="shared",
        default_ablation=ablation,
        run_tag=run_tag,
    )

    grid = {k: v for k, v in grid.items() if v}
    if not grid:
        raise ValueError("Grid search requested, but no grid values were provided.")

    keys = list(grid.keys())
    trials = list(itertools.product(*[grid[k] for k in keys]))
    if limit is not None:
        trials = trials[: max(0, int(limit))]

    print(
        f"[GRID] {len(trials)} trials on dataset='{base['dataset_name']}', "
        f"seed={seed}, base_tag={base_tag}. "
        f"Grid search uses shared mode and ignores comparison-only MODEL_RUN_SETTINGS."
    )

    for i, combo in enumerate(trials):
        overrides = {k: v for k, v in zip(keys, combo)}
        tag = "gs" + str(i).zfill(3) + "__" + "_".join([f"{k}-{_fmt(v)}" for k, v in overrides.items()])

        train_full_experiment(
            m_config_path=m_config_path,
            t_config_path=t_config_path,
            results_root=results_root,
            seed=seed,
            debug=debug,
            ablation=ablation,
            extra_tag=tag,
            overrides=overrides,
            run_mode="shared",
            run_tag=run_tag,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run CounterNet + CF pipeline on the dataset")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=PROJECT_ROOT / "assets" / "configs" / "home.json",
        help="Path to dataset/model config JSON (default: assets/configs/home.json)",
    )
    parser.add_argument(
        "--dataset-configs",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Run multiple datasets by providing multiple config JSON paths. "
            "Example: --dataset-configs assets/configs/adult.json assets/configs/credit_card.json"
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["adult", "credit_card", "home", "student"],
        default=None,
        help="Run a predefined set of datasets (mapped to assets/configs/<name>.json).",
    )
    parser.add_argument(
        "--trainer-config",
        type=Path,
        default=PROJECT_ROOT / "assets" / "configs" / "trainer.json",
        help="Path to trainer config JSON (default: assets/configs/trainer.json)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "assets" / "results",
        help="Root directory for experiment outputs and checkpoints (default: assets/results)",
    )
    parser.add_argument("--seed", type=int, default=31, help="Random seed (default: 31)")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="If set, train a new run. If not set, load a checkpoint instead.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Path to a run directory containing checkpoints. "
            "If omitted, uses --results-root/<run-tag>/<dataset>/seed-<seed>/"
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode: use only a few samples.")
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=["cfgen_neg_only", "cfgen_validity_neg_only", "cfgen_flip_neg_stay_pos"],
        help=(
            "Default ablation for models that do not set their own"
            "cfgen_neg_only: only use negative (result=0) examples to train the CF generator. "
            "cfgen_validity_neg_only: use all examples for CF proximity, but only enforce label-flip objective on negatives. "
            "cfgen_flip_neg_stay_pos: flip negatives, keep positives."
        ),
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["shared", "comparison"],
        default="shared",
        help=(
            "shared: all models use the same config"
            "comparison: each model can use its own overrides in MODEL_RUN_SETTINGS"
        ),
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Results tag"
        ),
    )

    # Grid search
    parser.add_argument("--grid", action="store_true", help="Run a small hyperparameter grid search (implies --retrain).")
    parser.add_argument("--grid_lambda2", type=str, default=None,
                        help="Comma-separated lambda_2 values (e.g. '0.001,0.01,0.05').")
    parser.add_argument("--grid_action_cost_base", type=str, default=None,
                        help="Comma-separated action_cost_base values (e.g. '0.005,0.01').")
    parser.add_argument("--grid_tau_mask", type=str, default=None,
                        help="Comma-separated gumbel_tau_mask values.")
    parser.add_argument("--grid_tau_choice", type=str, default=None,
                        help="Comma-separated gumbel_tau_choice values.")
    parser.add_argument("--grid_limit", type=int, default=None,
                        help="Optional cap on number of grid trials.")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    dataset_config_paths = resolve_dataset_config_paths(
        dataset_config=args.dataset_config,
        dataset_configs=args.dataset_configs,
        datasets=args.datasets,
    )

    if args.ckpt is not None and len(dataset_config_paths) > 1:
        raise ValueError(
            "--ckpt is only supported when running a single dataset"
        )

    if args.grid and args.retrain is False:
        print("[GRID] --grid implies retraining.")

    for m_config_path in dataset_config_paths:
        if args.grid:
            if args.ckpt is not None:
                raise ValueError("--grid is incompatible with --ckpt")

            grid: Dict[str, List[float]] = {
                "lambda_2": _parse_csv_floats(args.grid_lambda2),
                "action_cost_base": _parse_csv_floats(args.grid_action_cost_base),
            }
            tau_mask = _parse_csv_floats(args.grid_tau_mask)
            if tau_mask:
                grid["gumbel_tau_mask"] = tau_mask
            tau_choice = _parse_csv_floats(args.grid_tau_choice)
            if tau_choice:
                grid["gumbel_tau_choice"] = tau_choice

            run_grid_search(
                m_config_path=m_config_path,
                t_config_path=args.trainer_config,
                results_root=args.results_root,
                seed=args.seed,
                debug=args.debug,
                ablation=args.ablation,
                grid=grid,
                limit=args.grid_limit,
                run_tag=args.run_tag,
            )
            continue

        if args.retrain:
            train_full_experiment(
                m_config_path=m_config_path,
                t_config_path=args.trainer_config,
                results_root=args.results_root,
                seed=args.seed,
                debug=args.debug,
                ablation=args.ablation,
                run_mode=args.run_mode,
                run_tag=args.run_tag,
            )
        else:
            eval_from_checkpoint(
                m_config_path=m_config_path,
                t_config_path=args.trainer_config,
                results_root=args.results_root,
                seed=args.seed,
                run_dir_arg=args.ckpt,
                debug=args.debug,
                ablation=args.ablation,
                run_mode=args.run_mode,
                run_tag=args.run_tag,
            )


if __name__ == "__main__":
    main(sys.argv[1:])

    # Examples:
    # python scripts/run_home_counternet.py --retrain --datasets adult credit_card home student --ablation cfgen_flip_neg_stay_pos
    # python scripts/run_home_counternet.py --run-mode comparison --run-tag home_compare --retrain --datasets home
    # python scripts/run_home_counternet.py --grid --datasets home --grid_lambda2 0.001,0.01 --grid_action_cost_base 0.005,0.01

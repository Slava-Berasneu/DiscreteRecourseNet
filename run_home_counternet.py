from pathlib import Path
import argparse
import sys
import itertools
import shutil
from typing import Optional, List, Dict, Any

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from pytorch_lightning import seed_everything

from counternet.dataset import load_configs
from counternet.model import CounterNetModel, BaselinePredictiveModel, DiscreteRecourseNetModel
from counternet.pipeline import (
    Experiment,
    GlobalCFGenerator,
    LocalCFGenerator,
    Evaluator,
    ModelTrainer,
    load_trained_model,
)
from counternet.cf_explainer import VanillaCF


# Choose which CF generators to run.
MODELS_TO_RUN: List[str] = [
    "discrete_recoursenet",
]

MODEL_REGISTRY = {
    "counternet": CounterNetModel,
    "discrete_recoursenet": DiscreteRecourseNetModel,
    "vanillacf": VanillaCF,
}


def resolve_explainers(models_to_run: List[str]) -> List[type]:
    explainers: List[type] = []
    for name in models_to_run:
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Options: {sorted(MODEL_REGISTRY.keys())}")
        explainers.append(MODEL_REGISTRY[name])
    return explainers


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
            p_main = Path("assets/configs") / f"{name}.json"
            p_extra = Path("assets/configs/extra") / f"{name}.json"
            resolved.append(p_main if p_main.exists() else p_extra if p_extra.exists() else p_main)
        return resolved

    return [Path(dataset_config)]


def apply_ablation_to_config(m_config: dict, ablation: Optional[str]) -> dict:
    """Model ablation options"""
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

    m_config["ablation_tag"] = ablation
    return m_config


def resolve_run_dir(
    run_dir_arg: Optional[str],
    *,
    m_config: dict,
    results_root: Path,
    seed: int,
    ablation_tag: str = "default",
) -> Path:
    """<results_root>/<ablation_tag>/<dataset_name>/seed-<seed>/"""
    if run_dir_arg:
        p = Path(run_dir_arg)
        if not p.exists():
            raise FileNotFoundError(f"Run directory does not exist: {p}")
        if not p.is_dir():
            raise ValueError(f"--ckpt must point to a *run directory* (not a file). Got: {p}")
        return p

    return results_root / ablation_tag / m_config["dataset_name"] / f"seed-{seed}"


def find_checkpoint_in_run_dir(run_dir: Path, *, model_name: str) -> Path:
    checkpoint = run_dir / f"best_{model_name}.ckpt"
    if checkpoint.is_file():
        return checkpoint
    raise FileNotFoundError(
        f"best_{model_name}.ckpt checkpoint file not found. "
        f"Try running with --retrain first."
    )


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
    for pat in ("metrics.csv", "recourse_examples.csv", "*_results.pt", "*_actionability_groups.csv"):
        for p in dir_path.glob(pat):
            try:
                p.unlink()
            except Exception:
                pass


def _run_experiment(*, m_config: Dict[str, Any], t_config: Dict[str, Any],
                    results_root: Path, seed: int, debug: bool) -> None:
    selected = resolve_explainers(MODELS_TO_RUN)
    experiment = Experiment(
        explainers=selected,
        m_configs=[m_config],
        t_configs=t_config,
        debug=debug,
        results_root=results_root,
    )
    experiment.run(seeds=[seed])


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
) -> None:
    # Train + generate CFs + eval. Clean existing seed dir to avoid stale artifacts.
    m_config = load_configs(m_config_path)
    t_config = load_configs(t_config_path)

    m_config = apply_ablation_to_config(m_config, ablation)
    if overrides:
        m_config.update(overrides)

    ablation_tag = resolve_ablation_tag(m_config, ablation)
    if extra_tag:
        ablation_tag = f"{ablation_tag}__{extra_tag}"
        m_config["ablation_tag"] = ablation_tag

    run_dir = results_root / ablation_tag / m_config["dataset_name"] / f"seed-{seed}"
    _clean_run_dir(run_dir, keep_ckpts=False)

    print(
        f"[TRAIN] Running Experiment on dataset='{m_config['dataset_name']}', "
        f"seed={seed}, ablation={ablation_tag}"
    )

    _run_experiment(m_config=m_config, t_config=t_config, results_root=results_root, seed=seed, debug=debug)

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(run_dir.glob("*.ckpt"))
    if ckpts:
        print(f"[TRAIN] Saved checkpoints in {run_dir}:")
        for c in ckpts:
            print(f"  - {c}")
    else:
        print(f"[WARN] No .ckpt files found in {run_dir} (did ModelCheckpoint fire?)")


def eval_from_checkpoint(
    m_config_path: Path,
    t_config_path: Path,
    results_root: Path,
    seed: int,
    run_dir_arg: Optional[str],
    debug: bool,
    ablation: Optional[str],
) -> None:
    # Evaluate a previously trained run: regenerate CFs + compute metrics.
    seed_everything(seed, workers=True)

    m_config = load_configs(m_config_path)
    t_config = load_configs(t_config_path)

    m_config = apply_ablation_to_config(m_config, ablation)
    ablation_tag = resolve_ablation_tag(m_config, ablation)

    selected = resolve_explainers(MODELS_TO_RUN)

    # Train a baseline predictive model if we need VanillaCF
    pred_model = None
    if "vanillacf" in MODELS_TO_RUN:
        print("[EVAL] Training baseline predictive model for local explanations...")
        pred_model = BaselinePredictiveModel(m_config)
        pred_trainer = ModelTrainer(pred_model, t_config, logger_name="pred_model")
        pred_trainer.fit()

    run_dir = resolve_run_dir(
        run_dir_arg,
        m_config=m_config,
        results_root=results_root,
        seed=seed,
        ablation_tag=ablation_tag,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _clean_run_dir(run_dir, keep_ckpts=True)

    evaluator = Evaluator(configs={"is_logging": True})

    for explainer_cls in selected:
        name = explainer_cls.__name__

        if explainer_cls is VanillaCF:
            if pred_model is None:
                raise RuntimeError("VanillaCF requested but baseline predictive model was not created.")
            print("[EVAL] Generating local counterfactuals with VanillaCF...")
            local_cfg = dict(m_config)
            local_cfg.pop("cf_target_filter", None)
            local_cfg["lr"] = 0.05
            local_cfg["max_iter"] = 1000

            local_cf_gen = LocalCFGenerator(
                VanillaCF(pred_model.predict),
                pred_model,
                configs=local_cfg,
                ref_model=pred_model,
            )
            evaluator.eval(local_cf_gen.generate(debug=debug), run_dir)
            continue

        ckpt_path = find_checkpoint_in_run_dir(run_dir, model_name=name)
        print(f"[EVAL] Loading {name} checkpoint from: {ckpt_path}")
        global_model = explainer_cls(m_config)
        global_model.prepare_data()
        global_model = load_trained_model(global_model, checkpoint_path=str(ckpt_path), gpus=0)

        print(f"[EVAL] Generating global counterfactuals with {name}...")
        global_cf_gen = GlobalCFGenerator(
            global_model,
            configs=m_config,
            ref_model=pred_model if pred_model is not None else global_model,
        )
        evaluator.eval(global_cf_gen.generate(debug=debug), run_dir)

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
) -> None:
    base = load_configs(m_config_path)
    base = apply_ablation_to_config(base, ablation)
    base_tag = resolve_ablation_tag(base, ablation)

    keys = list(grid.keys())
    trials = list(itertools.product(*[grid[k] for k in keys]))
    if limit is not None:
        trials = trials[: max(0, int(limit))]

    print(f"[GRID] {len(trials)} trials on dataset='{base['dataset_name']}', seed={seed}, base_tag={base_tag}")

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
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run CounterNet + CF pipeline on the dataset")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("assets/configs/home.json"),
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
        default=Path("assets/configs/trainer.json"),
        help="Path to trainer config JSON (default: assets/configs/trainer.json)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("assets/results"),
        help="Root directory for experiment outputs and copied checkpoints (default: assets/results)",
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
            "Path to a *run directory* containing checkpoints. "
            "If omitted, uses --results-root/<ablation-tag>/<dataset>/seed-<seed>/"
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode: use only a few samples.")
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=["cfgen_neg_only", "cfgen_validity_neg_only", "cfgen_flip_neg_stay_pos"],
        help=(
            "cfgen_neg_only: only use negative (result=0) examples to train the CF generator. "
            "cfgen_validity_neg_only: use all examples for CF proximity, but only enforce label-flip objective on negatives. "
            "cfgen_flip_neg_stay_pos: flip negatives, keep positives."
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
            "--ckpt is only supported when running a single dataset. "
            "For multiple datasets, omit --ckpt and let the script pick checkpoints from --results-root."
        )

    for m_config_path in dataset_config_paths:
        if args.grid:
            if args.ckpt is not None:
                raise ValueError("--grid is incompatible with --ckpt (grid creates new run directories).")

            grid: Dict[str, List[float]] = {
                # Defaults
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
            )


if __name__ == "__main__":
    main(sys.argv[1:])

    # Examples:
    # python run_home_counternet.py --retrain --datasets adult credit_card home student
    # python run_home_counternet.py --datasets adult credit_card home student
    # python run_home_counternet.py --retrain --datasets adult credit_card home student --ablation cfgen_neg_only
    # python run_home_counternet.py --grid --datasets home --grid_lambda2 0.001,0.01 --grid_action_cost_base 0.005,0.01

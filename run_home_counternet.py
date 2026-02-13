from pathlib import Path
import argparse
import sys
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

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
from pytorch_lightning import seed_everything
from typing import Optional, List

# Choose which CF generators to run.
# Example: run DiscreteRecourseNet + CounterNet
# MODELS_TO_RUN = ["discrete_recoursenet", "counternet"]

MODELS_TO_RUN: List[str] = [
    "discrete_recoursenet",
    "counternet"
]

MODEL_REGISTRY = {
    "counternet": CounterNetModel,
    "discrete_recoursenet": DiscreteRecourseNetModel,
    "vanillacf": VanillaCF,
}


def resolve_explainers(models_to_run: List[str]) -> List[type]:
    explainers = []
    for name in models_to_run:
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Options: {sorted(MODEL_REGISTRY.keys())}")
        cls = MODEL_REGISTRY[name]
        explainers.append(cls)
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
    """
    Resolve which dataset config JSON files to run.
    """
    if dataset_configs:
        return [Path(p) for p in dataset_configs]

    if datasets:
        resolved_paths = []
        for name in datasets:
            # main config directory
            p_main = Path("assets/configs") / f"{name}.json"
            # extra directory
            p_extra = Path("assets/configs/extra") / f"{name}.json"

            if p_main.exists():
                resolved_paths.append(p_main)
            elif p_extra.exists():
                resolved_paths.append(p_extra)
            else:
                resolved_paths.append(p_main)
        return resolved_paths

    return [Path(dataset_config)]

def apply_ablation_to_config(m_config: dict, ablation: Optional[str]) -> dict:
    """
    Model ablation options
    """
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
            raise ValueError(
                f"--ckpt must point to a *run directory* (not a file). Got: {p}"
            )
        return p

    dataset_name = m_config["dataset_name"]
    return results_root / ablation_tag / dataset_name / f"seed-{seed}"


def find_checkpoint_in_run_dir(run_dir: Path, *, model_name: str) -> Path:
    """Find a model checkpoint inside a run directory."""

    checkpoint = run_dir / f"best_{model_name}.ckpt"
    if checkpoint.is_file():
        return checkpoint
    else:
        raise FileNotFoundError(
            f"best_{model_name}.ckpt checkpoint file not found"
            f"Try running with --retrain first."
        )

def train_full_experiment(m_config_path: Path, t_config_path: Path,
                          results_root: Path, seed: int, debug: bool,
                          ablation: Optional[str]) -> None:
    """Train CounterNet + baseline + run CF experiment"""
    m_config = load_configs(m_config_path)
    t_config = load_configs(t_config_path)

    m_config = apply_ablation_to_config(m_config, ablation)
    ablation_tag = resolve_ablation_tag(m_config, ablation)

    print(
        f"[TRAIN] Running Experiment on dataset='{m_config['dataset_name']}', "
        f"seed={seed}, ablation={ablation_tag}"
    )
    selected = resolve_explainers(MODELS_TO_RUN)
    experiment = Experiment(
        explainers=selected,
        m_configs=[m_config],
        t_configs=t_config,
        debug=debug,
        results_root=results_root
    )
    experiment.run(seeds=[seed])

    run_dir = results_root / ablation_tag / m_config["dataset_name"] / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(run_dir.glob("*.ckpt"))
    if ckpts:
        print(f"[TRAIN] Saved CounterNet checkpoints in {run_dir}:")
        for c in ckpts:
            print(f"  - {c}")
    else:
        print(f"[WARN] No .ckpt files found in {run_dir} (did ModelCheckpoint fire?)")


def eval_from_checkpoint(m_config_path: Path, t_config_path: Path,
                         results_root: Path, seed: int,
                         run_dir_arg: Optional[str], debug: bool,
                         ablation: Optional[str]) -> None:
    """Evaluate a previously trained run.
    Regenerate CFs + compute metrics for listed models
    """
    seed_everything(seed, workers=True)

    m_config = load_configs(m_config_path)
    t_config = load_configs(t_config_path)

    m_config = apply_ablation_to_config(m_config, ablation)
    ablation_tag = resolve_ablation_tag(m_config, ablation)
    dataset_name = m_config["dataset_name"]

    selected = resolve_explainers(MODELS_TO_RUN)

    # Train a baseline predictive model if we need VanillaCF
    pred_model = None
    if "vanillacf" in MODELS_TO_RUN:
        print(f"[EVAL] Training baseline predictive model for local explanations...")
        pred_model = BaselinePredictiveModel(m_config)
        pred_trainer = ModelTrainer(pred_model, t_config, logger_name="pred_model")
        pred_trainer.fit()

    # Resolve run directory and evaluator
    run_dir = resolve_run_dir(
        run_dir_arg,
        m_config=m_config,
        results_root=results_root,
        seed=seed,
        ablation_tag=ablation_tag,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(configs={"is_logging": True})

    # Evaluate each selected model
    for explainer_cls in selected:
        name = explainer_cls.__name__

        if explainer_cls is VanillaCF:
            if pred_model is None:
                raise RuntimeError("VanillaCF requested but baseline predictive model was not created.")
            print("[EVAL] Generating local counterfactuals with VanillaCF...")
            local_cfg = dict(m_config)
            local_cfg.pop("cf_target_filter", None)
            local_cfg['lr'] = 0.05
            local_cfg['max_iter'] = 1000

            local_cf_gen = LocalCFGenerator(
                VanillaCF(pred_model.predict),
                pred_model,
                configs=local_cfg,
                ref_model=pred_model,
            )
            local_results = local_cf_gen.generate(debug=debug)
            evaluator.eval(local_results, run_dir)
            continue

        # Global CF model: load its checkpoint and generate CFs
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
        global_results = global_cf_gen.generate(debug=debug)
        evaluator.eval(global_results, run_dir)

    print(f"[DONE] Evaluation finished. See {run_dir / 'metrics.csv'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run CounterNet + CF pipeline on the dataset"
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("assets/configs/home.json"),
        help="Path to dataset/model config JSON (default: assets/configs/home.json)",
    )
    parser.add_argument(
        "--dataset-configs",
        type = Path,
        nargs = "+",
        default = None,
        help = (
            "Run multiple datasets by providing multiple config JSON paths."
            "Example: --dataset-configs assets/configs/adult.json assets/configs/credit_card.json"
        ),
    )
    parser.add_argument(
"--datasets",
        type = str,
        nargs = "+",
        choices = ["adult", "credit_card", "home", "student"],
        default = None,
        help = (
            "Run a predefined set of datasets (mapped to assets/configs/<name>.json)."
            "Choices: adult, credit_card, home, student"
        ),
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
        help="Root directory for experiment outputs and copied checkpoints "
             "(default: assets/results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=31,
        help="Random seed used for Experiment directory naming (default: 31)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="If set, retrain CounterNet (and baseline) and save new checkpoints. "
             "If not set, load a checkpoint instead.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Path to a *run directory* containing checkpoints (e.g. assets/results/<ablation-tag>/<dataset>/seed-<SEED>/). "
            "If omitted, the script uses that default convention under --results-root."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: CF generators only run on a few samples.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=["cfgen_neg_only", "cfgen_validity_neg_only", "cfgen_flip_neg_stay_pos"],
        help=("cfgen_neg_only: only use negative (result=0) examples to train the CF generator."
              "cfgen_validity_neg_only: use all examples for CF proximity, but only enforce the label-flip objective on negatives"
              "cfgen_flip_neg_stay_pos: CF flip negatives, not flip non-negatives"
        ),
    )
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
            "--ckpt is only supported when running a single dataset."
            "For multiple datasets, omit --ckpt and let the script pick per-dataset checkpoints"
            "from --results-root/<ablation-tag>/<dataset>/seed-<seed>/, or run with --retrain."
        )

    for m_config_path in dataset_config_paths:
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
    # training example usage
    # python run_home_counternet.py --retrain --datasets adult credit_card home student

    # reusing checkpoints
    # python run_home_counternet.py --datasets adult credit_card home student

    # ablation
    # python run_home_counternet.py --retrain --datasets adult credit_card home student --ablation cfgen_neg_only
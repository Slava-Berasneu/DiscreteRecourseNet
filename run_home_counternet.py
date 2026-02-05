from pathlib import Path
import argparse
import sys
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from counternet.dataset import load_configs
from counternet.model import CounterNetModel, BaselinePredictiveModel
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

def find_checkpoint(ckpt_arg: str, m_config: dict, results_root: Path, seed: int, ablation_tag: str = "default") -> Path:
    """
    Searches for a model checkpoint and returns the path
    Prefers the best checkpoint
    """
    if ckpt_arg:
        ckpt_path = Path(ckpt_arg)
        if ckpt_path.is_file():
            return ckpt_path
        elif ckpt_path.is_dir():
            candidates = sorted(ckpt_path.glob("*.ckpt"))
            if not candidates:
                raise FileNotFoundError(f"No .ckpt files found in {ckpt_path}")
            return candidates[-1]
        else:
            raise FileNotFoundError(f"{ckpt_path} does not exist")

    dataset_name = m_config["dataset_name"]
    run_dir = results_root / ablation_tag / dataset_name / f"seed-{seed}"

    # Prefer the best.ckpt if it exists
    best_alias = run_dir / "best.ckpt"
    if best_alias.is_file():
        return best_alias

    candidates = sorted(run_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt files found in default results dir: {run_dir}\n"
            f"Try running with --retrain first, or pass --ckpt explicitly."
        )
    return candidates[-1]

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
    experiment = Experiment(
        explainers=[CounterNetModel, VanillaCF],
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
                         ckpt_arg: str, debug: bool,
                         ablation: Optional[str]) -> None:
    """
    Load CounterNet checkpoint and re-run:
      - global CFs (CounterNet as global explainer)
      - local CFs (VanillaCF on baseline predictive model)
    """
    seed_everything(seed, workers=True)

    m_config = load_configs(m_config_path)
    t_config = load_configs(t_config_path)

    m_config = apply_ablation_to_config(m_config, ablation)
    ablation_tag = resolve_ablation_tag(m_config, ablation)
    dataset_name = m_config["dataset_name"]

    # 1) Train a baseline predictive model for VanillaCF
    print(f"[EVAL] Training baseline predictive model for local explanations...")
    pred_model = BaselinePredictiveModel(m_config)
    pred_trainer = ModelTrainer(pred_model, t_config, logger_name="pred_model")
    pred_trainer.fit()

    # 2) Load CounterNet from checkpoint
    ckpt_path = find_checkpoint(ckpt_arg, m_config, results_root, seed, ablation_tag)
    print(f"[EVAL] Loading CounterNet checkpoint from: {ckpt_path}")
    cfnet_model = CounterNetModel(m_config)
    cfnet_model.prepare_data()
    cfnet_model = load_trained_model(cfnet_model, checkpoint_path=str(ckpt_path), gpus=0)

    # 3) Set up output dir and evaluator
    run_dir = results_root / ablation_tag / dataset_name / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(configs={"is_logging": True})

    # 4) Global CFs with CounterNet
    print("[EVAL] Generating global counterfactuals with CounterNet...")
    global_cf_gen = GlobalCFGenerator(
        cfnet_model,
        configs = m_config,
        ref_model = pred_model,
    )
    global_results = global_cf_gen.generate(debug=debug)
    evaluator.eval(global_results, run_dir)
    print("[EVAL] Global CF metrics updated in metrics.csv")

    # 5) Local CFs with VanillaCF + baseline predictive model
    print("[EVAL] Generating local counterfactuals with VanillaCF...")
    local_cfg = dict(m_config)
    local_cfg.pop("cf_target_filter", None)

    local_cfg['lr'] = 0.05
    local_cfg['max_iter'] = 1000

    local_cf_gen = LocalCFGenerator(
        VanillaCF(pred_model.predict),
        pred_model,
        configs = local_cfg,
        ref_model = pred_model,
    )
    local_results = local_cf_gen.generate(debug=debug)
    evaluator.eval(local_results, run_dir)
    print("[EVAL] Local CF metrics updated in metrics.csv")

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
        help="Path to a CounterNet .ckpt file OR a directory containing .ckpt files. "
             "If omitted, will look under assets/results/<ablation-tag>/<dataset>/seed-<SEED>/.",
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
                ckpt_arg=args.ckpt,
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
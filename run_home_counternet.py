from pathlib import Path
import argparse
import sys

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


def find_checkpoint(ckpt_arg: str, m_config: dict, results_root: Path, seed: int) -> Path:
    """
    Searches for a model checkpoint and returns the path
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
    run_dir = results_root / dataset_name / f"seed-{seed}"
    candidates = sorted(run_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt files found in default results dir: {run_dir}\n"
            f"Try running with --retrain first, or pass --ckpt explicitly."
        )
    return candidates[-1]


def train_full_experiment(m_config_path: Path, t_config_path: Path,
                          results_root: Path, seed: int, debug: bool) -> None:
    """Train CounterNet + baseline + run CF experiment"""
    m_config = load_configs(m_config_path)
    t_config = load_configs(t_config_path)

    print(f"[TRAIN] Running Experiment on dataset='{m_config['dataset_name']}', seed={seed}")
    experiment = Experiment(
        explainers=[CounterNetModel, VanillaCF],
        m_configs=[m_config],
        t_configs=t_config,
        debug=debug,
    )
    experiment.run(seeds=[seed])

    run_dir = results_root / m_config["dataset_name"] / f"seed-{seed}"
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
                         ckpt_arg: str, debug: bool) -> None:
    """
    Load CounterNet checkpoint and re-run:
      - global CFs (CounterNet as global explainer)
      - local CFs (VanillaCF on baseline predictive model)
    """

    # 0) Align dataset split with training
    seed_everything(seed, workers=True)

    m_config = load_configs(m_config_path)
    t_config = load_configs(t_config_path)
    dataset_name = m_config["dataset_name"]

    # 1) Train a baseline predictive model for VanillaCF
    print(f"[EVAL] Training baseline predictive model for local explanations...")
    pred_model = BaselinePredictiveModel(m_config)
    pred_trainer = ModelTrainer(pred_model, t_config, logger_name="pred_model")
    pred_trainer.fit()

    # 2) Load CounterNet from checkpoint
    ckpt_path = find_checkpoint(ckpt_arg, m_config, results_root, seed)
    print(f"[EVAL] Loading CounterNet checkpoint from: {ckpt_path}")
    cfnet_model = CounterNetModel(m_config)
    cfnet_model.prepare_data()
    cfnet_model = load_trained_model(cfnet_model, checkpoint_path=str(ckpt_path), gpus=0)

    # 3) Set up output dir and evaluator
    run_dir = results_root / dataset_name / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(configs={"is_logging": True})

    # 4) Global CFs with CounterNet
    print("[EVAL] Generating global counterfactuals with CounterNet...")
    global_cf_gen = GlobalCFGenerator(cfnet_model)
    global_results = global_cf_gen.generate(debug=debug)
    evaluator.eval(global_results, run_dir)
    print("[EVAL] Global CF metrics updated in metrics.csv")

    # 5) Local CFs with VanillaCF + baseline predictive model
    print("[EVAL] Generating local counterfactuals with VanillaCF...")
    local_cf_gen = LocalCFGenerator(VanillaCF(pred_model.predict), pred_model)
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
             "If omitted, will look under assets/results/<dataset>/seed-<SEED>/.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: CF generators only run on a few samples.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.retrain:
        train_full_experiment(
            m_config_path=args.dataset_config,
            t_config_path=args.trainer_config,
            results_root=args.results_root,
            seed=args.seed,
            debug=args.debug,
        )
    else:
        eval_from_checkpoint(
            m_config_path=args.dataset_config,
            t_config_path=args.trainer_config,
            results_root=args.results_root,
            seed=args.seed,
            ckpt_arg=args.ckpt,
            debug=args.debug,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
    # training example usage
    # python run_home_counternet.py --retrain
    # python run_home_counternet.py --retrain --debug --results-root assets/results_debug

    # reusing checkpoints
    # python run_home_counternet.py
    # python run_home_counternet.py --debug --results-root assets/results

    # explicit checkpoint selection
    # python run_home_counternet.py --ckpt assets/results/home/seed-0/epoch=2-step=59.ckpt

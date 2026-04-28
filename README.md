# DiscreteRecourseNet: End-to-End Actionable Recourse in Discrete Action Spaces

This is the code repository of the paper
[DiscreteRecourseNet: End-to-End Actionable Recourse in Discrete Action Spaces]()
which builds on the work of
[CounterNet: End-to-End Training of Counterfactual Aware Predictions](https://birkhoffg.github.io/files/icml21_workshop/counternet_paper.pdf). The purpose of the repository is only for research and reproduction of the paper's results.

This project primarily leverages `Pytorch` and `Pytorch Lightning` for implementations of deep learning models. To install all the dependencies, you should run:

```
pip install -e .
```

The main experiment entry point is `scripts/run_models.py`. For example, to run the shared experiment setup across the four datasets:

```
python scripts/run_models.py --retrain --datasets adult credit_card home student --ablation cfgen_flip_neg_stay_pos
```

Other command examples:
```
python scripts/run_models.py --run-mode comparison --run-tag model_compare --retrain --datasets home adult credit_card student
```
```
python scripts/run_models.py --grid --datasets home --grid_lambda2 0.001,0.01 --grid_action_cost_base 0.005,0.01
```
Notes:
- `pip install` will only install cpu-version of  `pytorch`. If you want to use GPU-version of `pytorch`, please follow [pytorch's official instruction](https://pytorch.org/get-started/locally/).
- As `Pytorch Lightning`'s API changes rapidly, it is not guaranteed that the code is compatible with other versions of Lightning (except the version that specified `settings.ini`).


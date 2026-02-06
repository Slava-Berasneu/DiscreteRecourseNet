# DiscreteRecourseNet: End-to-End Actionable Recourse in Discrete Action Spaces

This is the code repository of the paper
[DiscreteRecourseNet: End-to-End Actionable Recourse in Discrete Action Spaces]()
which serves as an extension of
[CounterNet: End-to-End Training of Counterfactual Aware Predictions](https://birkhoffg.github.io/files/icml21_workshop/counternet_paper.pdf). The purpose of the repository is only for research and reproduction of the paper's results. The audience should not expect to use the code directly in the deployed environemnt. 

This project primarily leverages `Pytorch` and `Pytorch Lightning` for implementations of deep learning models. To install all the dependencies, you should run:

```
pip install -e .
```

Notes:
- `pip install` will only install cpu-version of  `pytorch`. If you want to use GPU-version of `pytorch`, please follow [pytorch's official instruction](https://pytorch.org/get-started/locally/).
- As `Pytorch Lightning`'s API changes rapidly, it is not guaranteed that the code is compatible with other versions of Lightning (except the version that specified `settings.ini`).


__all__ = ['check_input_type', 'check_object_input_type', 'l1_mean', 'hinge_loss', 'get_loss_functions', 'split_X_y',
           'train_val_test_split', 'uniform', 'smooth_y', 'binarize_binary', 'flip_binary',
           'select_balanced_accuracy_threshold']

# Cell
from .import_essentials import *
import functools

# Cell
def _check_type(X):
    if not torch.is_tensor(X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        elif isinstance(X, list):
            X = torch.tensor(X)
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            X = torch.from_numpy(X)
        elif isinstance(X, pd.Series):
            X = X.values
            X = torch.tensor(X)
        else:
            raise ValueError(f'input X should be one of these types: [`list`, `pd.DataFrame`, `np.ndarray`, `torch.Tensor`], but got {type(X)}')
    return X.float()

# Cell
def check_input_type(func):
    """check if all inputs are torch.Tensor"""
    @functools.wraps(func)
    def wrapper_check_input_type(*args):
        new_args = []
        for X in list(args):
            new_args.append(_check_type(X))
        return func(*new_args)
    return wrapper_check_input_type

# Cell
def check_object_input_type(func):
    """check if all inputs are torch.Tensor"""
    @functools.wraps(func)
    def wrapper_check_input_type(ref, *args):
        new_args = [ref]
        for X in list(args):
            new_args.append(_check_type(X))
        return func(*new_args)
    return wrapper_check_input_type

# Comes from 02b_counter_net.ipynb, cell
def l1_mean(x, c):
    return F.l1_loss(x, c, reduction='mean') / x.abs().mean() # MAD

def hinge_loss(input, target):
    """
    reference:
    - https://github.com/interpretml/DiCE/blob/a772c8d4fcd88d1cab7f2e02b0bcc045dc0e2eab/dice_ml/explainer_interfaces/dice_pytorch.py#L196-L202
    - https://en.wikipedia.org/wiki/Hinge_loss
    """
    input = torch.log((abs(input - 1e-6) / (1 - abs(input - 1e-6))))
    all_ones = torch.ones_like(target)
    target = 2 * target - all_ones
    loss = all_ones - torch.mul(target, input)
    loss = F.relu(loss)
    return torch.norm(loss)

def get_loss_functions(f_name: str):
    _loss_functions = {
        'cross_entropy': F.binary_cross_entropy,
        'l1': F.l1_loss,
        'l1_mean': l1_mean,
        'mse': F.mse_loss
    }

    assert f_name in _loss_functions.keys(), \
        f'function name `{f_name}` is not in the loss function list {_loss_functions.keys()}'

    return _loss_functions[f_name]

# Comes from 02b_counter_net.ipynb, cell
def split_X_y(data: pd.DataFrame):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1]]
    return X, y

@check_input_type
def train_val_test_split(X, y):
    assert len(X) == len(y)
    size = len(X)
    train_size = int(0.7 * size)    # 70% for training
    val_size = int(0.8 * size)      # 10% for validation

    return (
        (X[: train_size], y[: train_size]),
        (X[train_size:val_size], y[train_size:val_size]),
        (X[val_size:], y[val_size:])
    )

# Comes from 02b_counter_net.ipynb, cell
def uniform(shape: tuple, r1: float, r2: float, device=None):
    assert r1 < r2, f"Issue: r1 ({r1}) >= r2 ({r2})"
    return (r2 - r1) * torch.rand(*shape, device=device) + r1


def smooth_y(y, device=None):
    return torch.where(y == 1,
        uniform(y.size(), 0.8, 0.95, device=y.device),
        uniform(y.size(), 0.05, 0.2, device=y.device))


def binarize_binary(x, threshold: Optional[float] = None, mode: str = "round"):
    x = _check_type(x)
    mode = str(mode or "round")
    assert ((x < 0) | (x > 1)).sum() == 0

    if mode == "round":
        return torch.round(x).clone().detach()

    if mode in ("fixed", "auto_val_balanced_accuracy"):
        thr = float(0.5 if threshold is None else threshold)
        return (x >= thr).to(x.dtype).clone().detach()

    raise ValueError(
        f"Unsupported prediction threshold mode '{mode}'. "
        "Expected one of: round, fixed, auto_val_balanced_accuracy."
    )


def flip_binary(x, threshold: Optional[float] = None, mode: str = "round"):
    x_bin = binarize_binary(x, threshold=threshold, mode=mode)
    return (1 - x_bin).clone().detach()


def select_balanced_accuracy_threshold(
    scores,
    target,
    *,
    default: float = 0.5,
) -> float:
    scores = _check_type(scores).reshape(-1).detach().cpu()
    target = _check_type(target).reshape(-1).detach().cpu()
    target = (target >= 0.5).to(torch.int64)

    if scores.numel() == 0 or target.numel() == 0:
        return float(default)

    pos = int((target == 1).sum().item())
    neg = int((target == 0).sum().item())
    if pos == 0 or neg == 0:
        return float(default)

    unique_scores = torch.unique(scores)
    if unique_scores.numel() == 0:
        return float(default)

    unique_scores, _ = torch.sort(unique_scores)
    eps = torch.finfo(unique_scores.dtype).eps
    candidates = torch.cat((unique_scores, unique_scores[-1:] + eps))

    best_ba = float("-inf")
    best_acc = float("-inf")
    best_distance = float("inf")
    best_threshold = float(default)

    for thr_t in candidates:
        thr = float(thr_t.item())
        pred = (scores >= thr_t).to(torch.int64)

        tp = int(((pred == 1) & (target == 1)).sum().item())
        tn = int(((pred == 0) & (target == 0)).sum().item())
        tpr = float(tp) / float(pos)
        tnr = float(tn) / float(neg)
        ba = 0.5 * (tpr + tnr)
        acc = float((pred == target).float().mean().item())
        distance = abs(thr - float(default))

        is_better = False
        if ba > best_ba + 1e-12:
            is_better = True
        elif abs(ba - best_ba) <= 1e-12 and acc > best_acc + 1e-12:
            is_better = True
        elif abs(ba - best_ba) <= 1e-12 and abs(acc - best_acc) <= 1e-12 and distance < best_distance - 1e-12:
            is_better = True

        if is_better:
            best_ba = ba
            best_acc = acc
            best_distance = distance
            best_threshold = thr

    return float(best_threshold)

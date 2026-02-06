__all__ = ['SensitivityMetric', 'proximity', 'ProximityMetric']

# Cell
from .import_essentials import *
from .utils import *

# Comes from 02b_counter_net.ipynb, cell
class SensitivityMetric(Metric):
    def __init__(self, predict_fn: Callable, scaler: ABCScaler, cat_idx: int, threshold: float):
        super().__init__(dist_sync_on_step=False)
        self.predict_fn = predict_fn
        self.scaler = scaler
        self.cat_idx = cat_idx
        self.threshold = threshold

        self.add_state("total_n_changes", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("diffs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, x: torch.Tensor, c: torch.Tensor, c_y: torch.Tensor):
        c_local = c.clone()

        x_cont_inv = self.scaler.inverse_transform(x[:, :self.cat_idx])
        c_cont_inv = self.scaler.inverse_transform(c_local[:, :self.cat_idx])

        cont_diff = torch.abs(x_cont_inv - c_cont_inv) < self.threshold
        mask = cont_diff.any(dim=1)  # samples with at least one small change

        n = mask.sum()
        if n == 0:
            return

        self.total_n_changes += n

        c_cont_hat = torch.where(cont_diff, x_cont_inv, c_cont_inv)
        c_local[:, :self.cat_idx] = self.scaler.transform(c_cont_hat)

        c_y_hat = self.predict_fn(c_local)

        self.diffs += (torch.round(c_y[mask]) != torch.round(c_y_hat[mask])).sum()


    def compute(self):
        if self.total_n_changes == 0:
            return torch.tensor(1.0, device=self.diffs.device)
        return 1 - self.diffs / self.total_n_changes


# Comes from 02b_counter_net.ipynb, cell
def proximity(x:torch.Tensor, c: torch.Tensor):
    return torch.abs(x - c).sum(dim=-1).mean()

# Comes from 02b_counter_net.ipynb, cell
class ProximityMetric(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("dist", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, x, c):
        self.dist += proximity(x, c)
        self.n += 1

    def compute(self):
        if self.n == 0:
            return -1
        else:
            return self.dist / self.n
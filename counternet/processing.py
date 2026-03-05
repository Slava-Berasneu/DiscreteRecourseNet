__all__ = ['ABCScaler', 'StandardScaler', 'MinMaxScaler', 'OneHotEncoder', 'NumpyDataset', 'PandasDataset',
           'CategoricalNormalizer']


from .import_essentials import *
from .functional_utils import *
from .dataset import load_adult_income_dataset


class ABCScaler(ABC):
    @abstractmethod
    def fit(self, X):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, X):
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, X):
        raise NotImplementedError


class StandardScaler(ABCScaler):
    """rewrite `StandardScaler` object in sci-kit learn in pytorch to eliminate cpu-gpu communication time"""
    mean_, std_ = None, None

    @check_object_input_type
    def fit(self, X):
        self.mean_, self.std_ = torch.mean(X), torch.std(X)
        return self

    @check_object_input_type
    def transform(self, X):
        if (self.mean_ is None) or (self.std_ is None):
            raise NotImplementedError(f'The scaler has not been fitted.')
        return (X - self.mean_) / self.std_

    @check_object_input_type
    def fit_transform(self, X):
        self.mean_, self.std_ = torch.mean(X), torch.std(X)
        return (X - self.mean_) / self.std_

    @check_object_input_type
    def inverse_transform(self, X):
        return X * self.std_ + self.mean_


class MinMaxScaler(ABCScaler):
    """Torch MinMaxScaler (global min/max) with sentinel handling.

    - If `special_values_` is set, those values are excluded from min/max fitting.
    - During transform, specials are mapped to fixed OOD coordinates (<= `ood_base`).
    - inverse_transform maps those OOD coordinates back to the original special values.

    - `min_` and `max_` are scalars computed over the entire continuous matrix, not per-feature
    """

    min_, max_ = None, None

    def __init__(self, special_values=None, ood_base: float = -1.0):
        self.special_values_ = list(special_values or [])
        self.ood_base = float(ood_base)
        self.special_to_ood = {}
        self.ood_to_special = {}
        if self.special_values_:
            self.set_special_values(self.special_values_, self.ood_base)

    def set_special_values(self, special_values, ood_base: float = -1.0):
        self.special_values_ = list(dict.fromkeys([float(v) for v in (special_values or [])]))
        self.ood_base = float(ood_base)
        # Deterministic OOD mapping: ood_base, ood_base-1, ood_base-2, ...
        self.special_to_ood = {sv: (self.ood_base - i) for i, sv in enumerate(self.special_values_)}
        self.ood_to_special = {ood: sv for sv, ood in self.special_to_ood.items()}
        return self

    def _fit_minmax_excluding_specials(self, X: torch.Tensor):
        if not self.special_values_:
            return torch.min(X), torch.max(X)

        flat = X.reshape(-1)
        sv = torch.tensor(self.special_values_, dtype=flat.dtype, device=flat.device)
        try:
            mask = ~torch.isin(flat, sv)
        except Exception:
            mask = torch.ones_like(flat, dtype=torch.bool)
            for v in sv:
                mask &= (flat != v)

        if mask.any():
            vals = flat[mask]
            return torch.min(vals), torch.max(vals)
        return torch.min(flat), torch.max(flat)

    @check_object_input_type
    def fit(self, X):
        self.min_, self.max_ = self._fit_minmax_excluding_specials(X)
        assert self.min_ != self.max_, f"min(X) == max(X) is not allowed."
        return self

    @check_object_input_type
    def transform(self, X):
        if (self.min_ is None) or (self.max_ is None):
            raise NotImplementedError(f'The scaler has not been fitted.')
        denom = (self.max_ - self.min_)
        scaled = (X - self.min_) / denom

        if self.special_values_:
            # keep exact constants for inverse_transform
            for sv, ood in self.special_to_ood.items():
                scaled = torch.where(
                    X == float(sv),
                    torch.tensor(float(ood), device=scaled.device, dtype=scaled.dtype),
                    scaled,
                )
        return scaled

    @check_object_input_type
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @check_object_input_type
    def inverse_transform(self, X):
        if (self.min_ is None) or (self.max_ is None):
            raise NotImplementedError(f'The scaler has not been fitted.')
        denom = (self.max_ - self.min_)
        raw = X * denom + self.min_

        if self.special_values_:
            for ood, sv in self.ood_to_special.items():
                raw = torch.where(
                    X == float(ood),
                    torch.tensor(float(sv), device=raw.device, dtype=raw.dtype),
                    raw,
                )
        return raw

class OneHotEncoder(object):
    categories_ = []
    drop_idx_ = None

    def __init__(self):
        from sklearn.preprocessing import OneHotEncoder
        self.enc = OneHotEncoder(sparse_output=False)

    def fit(self, X):
        self.enc.fit(X)
        # copy attributes
        self.categories_ = self.enc.categories_
        self.drop_idx_ = self.enc.drop_idx_
        return self

    def transform(self, X):
        return torch.from_numpy(self.enc.transform(X))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        assert isinstance(X, torch.Tensor)
        return self.enc.inverse_transform(X.cpu())


class NumpyDataset(TensorDataset):
    def __init__(self, *arrs):
        super().__init__()
        # init tensors
        # small patch: skip continous or discrete array without content
        self.tensors = [torch.tensor(arr).float()
                        for arr in arrs if arr.shape[-1] != 0]
        assert all(self.tensors[0].size(0) == tensor.size(0)
                   for tensor in self.tensors)

    def data_loader(self, batch_size=128, shuffle=True, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def features(self, test=False):
        return tuple(self.tensors[:-1] if not test else self.tensors)

    def target(self, test=False):
        return self.tensors[-1] if not test else None


class PandasDataset(NumpyDataset):
    def __init__(self, df: pd.DataFrame):
        cols = df.columns
        X = df[cols[:-1]].to_numpy()
        y = df[cols[-1]].to_numpy()
        super().__init__(X, y)


class CategoricalNormalizer(object):
    """implement post-processing step to enforce each elements
    in every category in the range of [0, 1] and output to 1.
    """
    def __init__(self, categories: List[List[Any]], cat_idx: int):
        self.categories = categories
        self.cat_idx = cat_idx

        # Build slices for each one-hot group inside the full feature vector x
        self.cat_slices = []
        start = cat_idx
        for cat in categories:
            end = start + len(cat)
            self.cat_slices.append((start, end))
            start = end

    def normalize(self, x, hard: bool = False):
        parts = []
        start = 0

        for cat_idx, cat_end_idx in self.cat_slices:
            if start < cat_idx:
                parts.append(x[:, start:cat_idx])

            logits = x[:, cat_idx:cat_end_idx]
            probs = torch.softmax(logits, dim=-1)

            if hard:
                idx = probs.argmax(dim=-1)
                onehot = torch.nn.functional.one_hot(
                    idx, num_classes=probs.shape[-1]
                ).to(probs.dtype)
                probs = probs + (onehot - probs).detach()

            parts.append(probs)
            start = cat_end_idx

        if start < x.shape[1]:
            parts.append(x[:, start:])

        return torch.cat(parts, dim=1)

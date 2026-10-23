"""CHELSA v2.1 variables, normalization, and training data loader."""
import os

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

VAR_NAMES = ["cmi", "clt", "hurs", "pet", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin", "vpd"]
UNITS = ["kg m-2 month-1", "%", "%", "kg m-2 month-1", "kg m-2 month-1", "MJ m-2 d-1", "m s-1",
         "degC", "degC", "degC", "Pa"]

# CLIMPLICIT normalization constants, in CHELSA's stored units.
MEAN = np.array([
    -264.1493656, 3912.44628016, 5921.65964573, 9385.47468266, 697.03653109,
    15219.37926928, 3498.8511804, 2819.56006368, 2864.08583811, 2773.46759638,
    8039.37322797,
])
STD = np.array([
    1042.67560332, 1767.94018571, 1185.91587823, 6639.79069994, 883.56243405,
    7843.49167037, 1637.09237995, 174.43791946, 181.69448751, 167.07485901,
    7516.98198719,
])
SCALE = np.array([0.1, 0.01, 0.01, 0.01, 0.1, 0.001, 0.001, 0.1, 0.1, 0.1, 0.1])
OFFSET = np.array([0, 0, 0, 0, 0, 0, 0, -273.15, -273.15, -273.15, 0])


def denormalize(y: torch.Tensor) -> torch.Tensor:
    """(..., 11) model output -> physical units (UNITS)."""
    a = torch.as_tensor(STD * SCALE, dtype=y.dtype, device=y.device)
    b = torch.as_tensor(MEAN * SCALE + OFFSET, dtype=y.dtype, device=y.device)
    return y * a + b


def land_split(n_land: int, val_frac: float = 0.02, seed: int = 42):
    """Deterministic train/val split of land-pixel indices."""
    g = np.random.default_rng(seed)
    perm = g.permutation(n_land)
    n_val = int(n_land * val_frac)
    return perm[n_val:], perm[:n_val]   # (train_idx, val_idx)


def load_month(numpy_dir, month, in_memory):
    path = os.path.join(numpy_dir, f"{month:02d}_monthly_float16_land_only.npy")
    arr = np.load(path, mmap_mode=None if in_memory else "r")
    return np.ascontiguousarray(arr) if in_memory else arr


class CHELSADataset(Dataset):
    """Returns a whole batch per __getitem__, one month per pixel (random for train, fixed for val)."""

    def __init__(self, point_to_coord, monthly, indices, months, random_month=True):
        self.ptc = point_to_coord           # (N_land, 2) memmap or array
        self.monthly = monthly              # {month: (11, N_land)}
        self.indices = np.asarray(indices)
        self.months = np.asarray(months)
        self.random_month = random_month

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        idx = self.indices[np.asarray(pos)]
        n = len(idx)
        if self.random_month:
            mo = self.months[torch.randint(len(self.months), (n,)).numpy()]
        else:
            mo = self.months[idx % len(self.months)]
        latlon = torch.from_numpy(np.ascontiguousarray(self.ptc[idx]).astype("float32"))
        target = np.empty((n, 11), dtype="float32")
        for m in np.unique(mo):                          # one gather per month
            sel = mo == m
            target[sel] = self.monthly[int(m)][:, idx[sel]].T
        return latlon, torch.from_numpy(mo.astype("int64")), torch.from_numpy(target)


def chelsa_loaders(numpy_dir, batch_size=8192, num_workers=8, val_frac=0.02, split_seed=42, in_memory=False):
    """(train_loader, val_loader)."""
    ptc = np.load(os.path.join(numpy_dir, "point_to_coord.npy"), mmap_mode=None if in_memory else "r")
    months = list(range(1, 13))
    monthly = {m: load_month(numpy_dir, m, in_memory) for m in months}
    train_idx, val_idx = land_split(ptc.shape[0], val_frac, split_seed)

    def loader(ds, shuffle):
        base = RandomSampler(ds) if shuffle else SequentialSampler(ds)
        return DataLoader(ds, sampler=BatchSampler(base, batch_size, drop_last=shuffle), batch_size=None,
                          num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    return (loader(CHELSADataset(ptc, monthly, train_idx, months, random_month=True), True),
            loader(CHELSADataset(ptc, monthly, val_idx, months, random_month=False), False))

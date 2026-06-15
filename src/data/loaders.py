"""
Dataset loaders.  Supports MNIST, FashionMNIST, NMNIST, DVS Gesture, and SHD.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def _flat_normalized_transform(mean: float, std: float):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
            transforms.Lambda(lambda x: x.view(-1)),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),
        ]
    )


class _TonicDataset(Dataset):
    """Generic tonic dataset wrapper: events → [T, C*H*W] float spike tensor."""

    def __init__(self, tonic_ds):
        self._ds = tonic_ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        frames, label = self._ds[idx]  # frames: np [T, C, H, W]
        frames = torch.tensor(frames, dtype=torch.float32)
        frames = frames.clamp(0, 1)  # event counts → binary spikes
        frames = frames.flatten(start_dim=1)  # [T, C*H*W]
        return frames, label


def _make_nmnist(root: str, train: bool, T: int):
    import tonic
    import tonic.transforms as tonic_transforms

    sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
    transform = tonic_transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)
    return _TonicDataset(
        tonic.datasets.NMNIST(save_to=root, train=train, transform=transform)
    )


def _make_dvs_gesture(root: str, train: bool, T: int):
    import tonic
    import tonic.transforms as tonic_transforms

    target_size = (32, 32, 2)
    transform = tonic_transforms.Compose(
        [
            tonic_transforms.Downsample(spatial_factor=0.25),  # 128×128 → 32×32
            tonic_transforms.ToFrame(sensor_size=target_size, n_time_bins=T),
        ]
    )
    return _TonicDataset(
        tonic.datasets.DVSGesture(save_to=root, train=train, transform=transform)
    )


class _SHDDataset(Dataset):
    """SHD dataset: spike events → (T, 700) binned tensor."""

    def __init__(
        self, tonic_ds, T: int, dt_us: float = 10_000.0, n_channels: int = 700
    ):
        self._ds = tonic_ds
        self.T = T
        self.dt_us = (
            dt_us  # bin width in microseconds (10ms = 10_000us) -> us: micro second
        )
        self.n_channels = n_channels

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        events, label = self._ds[idx]
        binned = torch.zeros(self.T, self.n_channels)
        if len(events) > 0:
            t = torch.from_numpy(events["t"].astype(np.int64))
            x = torch.from_numpy(events["x"].astype(np.int64))
            time_bins = (t / self.dt_us).long().clamp(0, self.T - 1)
            x = x.clamp(0, self.n_channels - 1)
            binned[time_bins, x] = 1.0
        return binned, label


def _make_shd(root: str, train: bool, T: int):
    import tonic

    return _SHDDataset(tonic.datasets.SHD(save_to=root, train=train), T=T)


def _make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def _split_train_validation(train_ds, cfg):
    use_validation = getattr(cfg, "use_validation", False)
    val_fraction = float(getattr(cfg, "val_fraction", 0.0))
    if not use_validation or val_fraction <= 0.0:
        return train_ds, None

    n_total = len(train_ds)
    if n_total < 2:
        raise ValueError("Validation split requires at least 2 training samples.")

    val_size = int(round(n_total * val_fraction))
    val_size = max(1, min(val_size, n_total - 1))
    train_size = n_total - val_size
    generator = torch.Generator().manual_seed(int(getattr(cfg, "val_seed", 42)))
    train_subset, val_subset = random_split(
        train_ds, [train_size, val_size], generator=generator
    )
    return train_subset, val_subset


def get_dataloaders(cfg) -> tuple:
    """Return (train_loader, test_loader) for the dataset specified in cfg."""
    dataset = cfg.dataset.lower()

    if dataset == "mnist":
        from torchvision import datasets

        transform = _flat_normalized_transform(0.1307, 0.3081)
        train_ds = datasets.MNIST(
            cfg.data_dir, train=True, download=True, transform=transform
        )
        test_ds = datasets.MNIST(
            cfg.data_dir, train=False, download=True, transform=transform
        )

    elif dataset == "fashion_mnist":
        from torchvision import datasets

        transform = _flat_normalized_transform(0.2860, 0.3530)
        train_ds = datasets.FashionMNIST(
            cfg.data_dir, train=True, download=True, transform=transform
        )
        test_ds = datasets.FashionMNIST(
            cfg.data_dir, train=False, download=True, transform=transform
        )

    elif dataset == "nmnist":
        train_ds = _make_nmnist(cfg.data_dir, train=True, T=cfg.T)
        test_ds = _make_nmnist(cfg.data_dir, train=False, T=cfg.T)

    elif dataset == "dvs_gesture":
        train_ds = _make_dvs_gesture(cfg.data_dir, train=True, T=cfg.T)
        test_ds = _make_dvs_gesture(cfg.data_dir, train=False, T=cfg.T)

    elif dataset == "shd":
        train_ds = _make_shd(cfg.data_dir, train=True, T=cfg.T)
        test_ds = _make_shd(cfg.data_dir, train=False, T=cfg.T)

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset!r}")

    train_loader = _make_loader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = _make_loader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader


def get_train_val_test_dataloaders(cfg) -> tuple:
    """Return (train_loader, val_loader, test_loader) for LSM-style training."""
    dataset = cfg.dataset.lower()
    if dataset != "shd":
        train_loader, test_loader = get_dataloaders(cfg)
        return train_loader, None, test_loader

    train_ds = _make_shd(cfg.data_dir, train=True, T=cfg.T)
    test_ds = _make_shd(cfg.data_dir, train=False, T=cfg.T)
    train_ds, val_ds = _split_train_validation(train_ds, cfg)

    train_loader = _make_loader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = (
        _make_loader(val_ds, batch_size=cfg.batch_size, shuffle=False)
        if val_ds is not None
        else None
    )
    test_loader = _make_loader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

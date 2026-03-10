"""Data streaming utilities for continual learning in KWS."""

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass
class ContinualDataStream:
    """Online stream over old/new class pools with configurable mixing.

    mix_old_ratio=0.75 means each sampled item is old-class with probability 0.75
    and new-class with probability 0.25.
    """

    x_old: np.ndarray
    y_old: np.ndarray
    x_new: np.ndarray
    y_new: np.ndarray
    mix_old_ratio: float
    batch_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.mix_old_ratio <= 1.0):
            raise ValueError("mix_old_ratio must be in [0, 1].")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.x_old.shape[0] == 0:
            raise ValueError("Old-class pool is empty.")
        if self.x_new.shape[0] == 0:
            raise ValueError("New-class pool is empty.")
        self.rng = np.random.default_rng(self.seed)
        self.total_old_drawn = 0
        self.total_new_drawn = 0
        self.per_class_draw_counts = {}

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample one batch in random order from mixed old/new draws."""
        choose_old = self.rng.random(self.batch_size) < self.mix_old_ratio
        x_batch = np.empty((self.batch_size,) + self.x_old.shape[1:], dtype=self.x_old.dtype)
        y_batch = np.empty((self.batch_size,), dtype=self.y_old.dtype)

        old_count = int(np.sum(choose_old))
        new_count = self.batch_size - old_count

        if old_count > 0:
            old_idx = self.rng.integers(0, self.x_old.shape[0], size=old_count)
            x_batch[choose_old] = self.x_old[old_idx]
            y_batch[choose_old] = self.y_old[old_idx]
            self.total_old_drawn += old_count

        if new_count > 0:
            new_idx = self.rng.integers(0, self.x_new.shape[0], size=new_count)
            x_batch[~choose_old] = self.x_new[new_idx]
            y_batch[~choose_old] = self.y_new[new_idx]
            self.total_new_drawn += new_count

        # Keep batch order random.
        perm = self.rng.permutation(self.batch_size)
        x_batch = x_batch[perm]
        y_batch = y_batch[perm]

        labels, counts = np.unique(y_batch, return_counts=True)
        for lbl, cnt in zip(labels.tolist(), counts.tolist()):
            self.per_class_draw_counts[int(lbl)] = self.per_class_draw_counts.get(int(lbl), 0) + int(cnt)
        return x_batch, y_batch

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        while True:
            yield self.sample_batch()

    def get_stream_stats(self) -> dict:
        total = self.total_old_drawn + self.total_new_drawn
        ratio_old = float(self.total_old_drawn / total) if total > 0 else float("nan")
        ratio_new = float(self.total_new_drawn / total) if total > 0 else float("nan")
        return {
            "total_drawn": int(total),
            "old_drawn": int(self.total_old_drawn),
            "new_drawn": int(self.total_new_drawn),
            "ratio_old": ratio_old,
            "ratio_new": ratio_new,
            "per_class_draw_counts": dict(sorted(self.per_class_draw_counts.items())),
        }


def build_stream_from_dataset(
    x: np.ndarray,
    y: np.ndarray,
    old_class_indices: np.ndarray,
    new_class_indices: np.ndarray,
    mix_old_ratio: float,
    batch_size: int,
    seed: int = 0,
    balance_per_class: bool = True,
) -> ContinualDataStream:
    """Split dataset into old/new pools and build continual stream."""
    old_mask = np.isin(y, old_class_indices)
    new_mask = np.isin(y, new_class_indices)

    x_old, y_old = x[old_mask], y[old_mask]
    x_new, y_new = x[new_mask], y[new_mask]

    if balance_per_class:
        x_old, y_old = rebalance_pool_by_class(x_old, y_old, seed=seed + 11)
        x_new, y_new = rebalance_pool_by_class(x_new, y_new, seed=seed + 17)

    return ContinualDataStream(
        x_old=x_old,
        y_old=y_old,
        x_new=x_new,
        y_new=y_new,
        mix_old_ratio=mix_old_ratio,
        batch_size=batch_size,
        seed=seed,
    )


def rebalance_pool_by_class(x: np.ndarray, y: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample each class in a pool to the same count (the class minimum)."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")
    if x.shape[0] == 0:
        raise ValueError("Cannot rebalance an empty pool.")

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    class_indices = {int(c): np.where(y == c)[0] for c in classes}
    min_count = min(int(idx.size) for idx in class_indices.values())
    if min_count < 1:
        raise ValueError("Found class with zero samples while rebalancing.")

    picked = []
    for c in classes:
        idx = class_indices[int(c)]
        choose = rng.permutation(idx)[:min_count]
        picked.append(choose)
    keep = np.concatenate(picked, axis=0)
    keep = keep[rng.permutation(keep.shape[0])]
    return x[keep], y[keep]

from dataclasses import replace
import os
from pathlib import Path
from typing import Dict, Tuple
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset

from kaf_profiti.industrial.missing import MissingMechanismSimulator


_LOAD_ERRORS = (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile)


def _load_cached_arrays(path: Path, expected_shapes: Dict[str, Tuple[int, ...]]):
    try:
        with np.load(path) as loaded:
            arrays = {}
            for key, shape in expected_shapes.items():
                if key not in loaded.files:
                    raise KeyError(key)
                array = loaded[key].astype(np.uint8)
                if array.shape != shape:
                    raise ValueError(f"cached mask {key} shape {array.shape} != {shape}")
                arrays[key] = array
            return arrays
    except _LOAD_ERRORS:
        return None


def _atomic_savez_compressed(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def generate_masks_array(
    num_windows: int,
    history_len: int,
    num_sensors: int,
    missing_rate: float,
    seed: int,
    mode: str = "mixed",
) -> np.ndarray:
    if not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate must be in [0, 1]")
    if missing_rate == 0 or mode == "none":
        return np.ones((num_windows, history_len, num_sensors), dtype=np.uint8)
    simulator = MissingMechanismSimulator(
        mode=mode,
        random_keep_prob=max(0.0, min(1.0, 1.0 - missing_rate)),
    )
    masks = np.empty((num_windows, history_len, num_sensors), dtype=np.uint8)
    for idx in range(num_windows):
        mask = simulator((history_len, num_sensors), seed + idx * 7919)
        masks[idx] = mask.to(torch.uint8).numpy()
    return masks


def generate_or_load_masks(
    path,
    num_windows: int,
    history_len: int,
    num_sensors: int,
    missing_rate: float,
    seed: int,
    mode: str = "mixed",
) -> np.ndarray:
    path = Path(path)
    if path.exists():
        loaded = _load_cached_arrays(
            path,
            {"mask": (num_windows, history_len, num_sensors)},
        )
        if loaded is not None:
            return loaded["mask"]
    masks = generate_masks_array(num_windows, history_len, num_sensors, missing_rate, seed, mode)
    _atomic_savez_compressed(
        path,
        mask=masks,
        missing_rate=float(missing_rate),
        seed=int(seed),
        mode=str(mode),
    )
    return masks


def generate_or_load_split_masks(
    path,
    split_shapes: Dict[str, Tuple[int, int, int]],
    missing_rate: float,
    seed: int,
    mode: str = "mixed",
) -> Dict[str, np.ndarray]:
    path = Path(path)
    if path.exists():
        loaded = _load_cached_arrays(path, split_shapes)
        if loaded is not None:
            return loaded
    arrays = {}
    for offset, (split, shape) in enumerate(split_shapes.items()):
        arrays[split] = generate_masks_array(
            shape[0],
            shape[1],
            shape[2],
            missing_rate,
            seed + offset * 100_003,
            mode,
        )
    _atomic_savez_compressed(
        path,
        **arrays,
        missing_rate=float(missing_rate),
        seed=int(seed),
        mode=str(mode),
    )
    return arrays


class MaskedWindowDataset(Dataset):
    def __init__(self, dataset: Dataset, masks: np.ndarray):
        if len(dataset) != len(masks):
            raise ValueError(f"dataset length {len(dataset)} != mask length {len(masks)}")
        self.dataset = dataset
        self.masks = masks.astype(np.uint8)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        mask = torch.tensor(self.masks[index], dtype=torch.float32)
        return replace(sample, M_obs=mask, X_obs=sample.X_obs * mask)

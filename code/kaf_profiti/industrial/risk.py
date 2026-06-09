from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor

from .cmapss import SENSOR_COLUMNS


def compute_sensor_thresholds(
    frame: pd.DataFrame,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    sensor_mean: Optional[Tensor] = None,
    sensor_std: Optional[Tensor] = None,
    sensor_columns: Optional[Sequence[str]] = None,
) -> Tuple[Tensor, Tensor]:
    columns = list(sensor_columns or SENSOR_COLUMNS)
    lower = torch.tensor(
        frame[columns].quantile(lower_quantile).to_numpy(), dtype=torch.float32
    )
    upper = torch.tensor(
        frame[columns].quantile(upper_quantile).to_numpy(), dtype=torch.float32
    )
    if sensor_mean is not None and sensor_std is not None:
        lower = (lower - sensor_mean) / sensor_std
        upper = (upper - sensor_mean) / sensor_std
    return lower, upper


def threshold_risk_score(samples: Tensor, upper_limits: Tensor, lower_limits: Tensor) -> Tensor:
    if samples.dim() != 4:
        raise ValueError("samples must have shape (B, S, Lp, N)")
    upper = upper_limits.to(samples.device).view(1, 1, 1, -1)
    lower = lower_limits.to(samples.device).view(1, 1, 1, -1)
    if upper.shape[-1] != samples.shape[-1] or lower.shape[-1] != samples.shape[-1]:
        raise ValueError("threshold dimension must match sensor dimension")
    crossed = (samples > upper) | (samples < lower)
    any_crossed = crossed.any(dim=(2, 3)).float()
    return any_crossed.mean(dim=1)


def threshold_exceedance_risk_score(samples: Tensor, upper_limits: Tensor, lower_limits: Tensor) -> Tensor:
    if samples.dim() != 4:
        raise ValueError("samples must have shape (B, S, Lp, N)")
    upper = upper_limits.to(samples.device).view(1, 1, 1, -1)
    lower = lower_limits.to(samples.device).view(1, 1, 1, -1)
    if upper.shape[-1] != samples.shape[-1] or lower.shape[-1] != samples.shape[-1]:
        raise ValueError("threshold dimension must match sensor dimension")
    crossed = (samples > upper) | (samples < lower)
    return crossed.float().mean(dim=(1, 2, 3))


def risk_from_samples(samples: Tensor, lower: Tensor, upper: Tensor) -> Dict[str, Tensor]:
    if samples.dim() != 4:
        raise ValueError("samples must have shape (B, S, Lp, N)")
    lower = lower.to(samples.device).view(1, 1, 1, -1)
    upper = upper.to(samples.device).view(1, 1, 1, -1)
    crossed = (samples < lower) | (samples > upper)
    sensor_risk = crossed.any(dim=2).float().mean(dim=1)
    device_risk = crossed.any(dim=(2, 3)).float().mean(dim=1)
    return {"sensor_risk": sensor_risk, "device_risk": device_risk}


def rul_to_state(rul: Tensor, warning_threshold: float = 30.0, critical_threshold: float = 15.0) -> Tensor:
    state = torch.zeros_like(rul, dtype=torch.long)
    state = torch.where(rul <= warning_threshold, torch.ones_like(state), state)
    state = torch.where(rul <= critical_threshold, torch.full_like(state, 2), state)
    return state

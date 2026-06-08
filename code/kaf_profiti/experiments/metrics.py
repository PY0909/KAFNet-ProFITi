from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

from kaf_profiti.industrial.risk import threshold_risk_score

try:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    f1_score = None
    roc_auc_score = None


def safe_binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5):
    labels = labels.astype(int)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": None, "auprc": None, "f1": None}
    pred = (scores >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)) if roc_auc_score else None,
        "auprc": float(average_precision_score(labels, scores)) if average_precision_score else None,
        "f1": float(f1_score(labels, pred)) if f1_score else None,
    }


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> Optional[float]:
    labels = labels.astype(float)
    if labels.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        in_bin = (scores >= start) & (scores < end if end < 1.0 else scores <= end)
        if not in_bin.any():
            continue
        confidence = scores[in_bin].mean()
        accuracy = labels[in_bin].mean()
        ece += float(in_bin.mean() * abs(confidence - accuracy))
    return ece


def interval_metrics(y: Tensor, samples: Tensor, mask: Tensor, alpha: float = 0.05):
    lower = torch.quantile(samples, alpha / 2, dim=1)
    upper = torch.quantile(samples, 1 - alpha / 2, dim=1)
    covered = ((y >= lower) & (y <= upper)).float() * mask
    picp = covered.sum() / mask.sum().clamp_min(1.0)
    mpiw = ((upper - lower) * mask).sum() / mask.sum().clamp_min(1.0)
    return float(picp.cpu()), float(mpiw.cpu())


def point_metrics(y: Tensor, mean: Tensor, mask: Tensor):
    diff = (mean - y) * mask
    mae = diff.abs().sum() / mask.sum().clamp_min(1.0)
    rmse = torch.sqrt((diff.pow(2).sum() / mask.sum().clamp_min(1.0)).clamp_min(0.0))
    return float(mae.cpu()), float(rmse.cpu())


def crps_from_samples(y: Tensor, samples: Tensor, mask: Tensor) -> Tensor:
    term1 = (samples - y.unsqueeze(1)).abs().mean(dim=1)
    pairwise = (samples.unsqueeze(2) - samples.unsqueeze(1)).abs().mean(dim=(1, 2))
    crps = (term1 - 0.5 * pairwise) * mask
    return crps.sum() / mask.sum().clamp_min(1.0)


def heuristic_risk_score_from_samples(samples: Tensor) -> Tensor:
    return torch.sigmoid(samples.abs().mean(dim=(1, 2, 3)))


risk_score_from_samples = heuristic_risk_score_from_samples


def completed_metrics_template() -> Dict[str, object]:
    return {
        "mae": None,
        "rmse": None,
        "nll": None,
        "crps": None,
        "picp": None,
        "mpiw": None,
        "auroc": None,
        "auprc": None,
        "f1": None,
        "ece": None,
        "lead_time": None,
        "train_time_sec": None,
        "infer_time_ms_per_batch": None,
        "num_params": None,
        "gpu_memory_mb": None,
    }

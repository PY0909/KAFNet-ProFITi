#!/usr/bin/env python
import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from kaf_profiti.experiments.datasets import create_protocol_datasets
from kaf_profiti.experiments.masks import MaskedWindowDataset, generate_or_load_split_masks
from kaf_profiti.experiments.metrics import (
    completed_metrics_template,
    crps_from_samples,
    expected_calibration_error,
    interval_metrics,
    point_metrics,
    safe_binary_metrics,
    threshold_risk_score,
)
from kaf_profiti.experiments.registry import create_model, get_model_spec
from kaf_profiti.industrial.batch import IndustrialCollator


@dataclass
class ExperimentConfig:
    dataset: str
    model: str = "kaf_profiti_joint"
    seed: int = 2026
    missing_rate: float = 0.3
    history_len: int = 96
    pred_len: int = 24
    stride: int = 1
    data_root: str = "dataset"
    output_dir: str = "code/results"
    epochs: int = 1
    batch_size: int = 16
    max_train_batches: int = 20
    max_eval_batches: int = 20
    nsamples: int = 20
    device: str = "cpu"
    missing_mode: str = "mixed"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    risk_threshold: float = 30.0
    hidden_dim: int = 32
    te_dim: int = 5
    kernel_count: int = 4
    n_layers: int = 2
    n_heads: int = 2
    flow_layers: int = 2
    preconv_dim: int = 8
    lambda_point: float = 0.1


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _write_config_yaml(path: Path, config: ExperimentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {value}" for key, value in asdict(config).items()]
    path.write_text("\n".join(lines) + "\n")


def _train_epoch(model, loader, optimizer, device: torch.device, max_batches: int):
    total = 0.0
    count = 0
    model.train()
    for idx, batch in enumerate(loader):
        if max_batches and idx >= max_batches:
            break
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = _model_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


def _model_loss(model, batch):
    if hasattr(model, "flow_head"):
        return model.loss(batch, nsamples_for_point=1)
    return model.loss(batch)


def _risk_labels(dataset: str, batch, risk_threshold: float):
    if dataset.startswith("cmapss"):
        return (batch.rul <= risk_threshold).float()
    return batch.rul.float().clamp(0, 1)


def _distribution_metrics(model, batch, nsamples: int):
    hidden = model.distribution(batch)
    if hasattr(model, "flow_head"):
        nll = model.flow_head.nll(batch.y_flat, hidden, batch.mq_flat).mean()
        samples = model.flow_head.sample(hidden, batch.mq_flat, nsamples=nsamples)
        crps = model.flow_head.crps(batch.y_flat, samples, batch.mq_flat)
        return nll, samples, crps
    if hasattr(model, "head"):
        nll = model.head.nll(batch.y_flat, hidden, batch.mq_flat).mean()
        samples = model.head.sample(hidden, batch.mq_flat, nsamples=nsamples)
        crps = crps_from_samples(batch.y_flat, samples, batch.mq_flat)
        return nll, samples, crps
    raise TypeError(f"Unsupported model type for evaluation: {type(model).__name__}")


def _evaluate(
    model,
    loader,
    device: torch.device,
    config: ExperimentConfig,
    num_sensors: int,
    risk_lower_limits: torch.Tensor,
    risk_upper_limits: torch.Tensor,
):
    totals = {"nll": 0.0, "mae": 0.0, "rmse": 0.0, "crps": 0.0, "picp": 0.0, "mpiw": 0.0}
    count = 0
    means: List[np.ndarray] = []
    samples_out: List[np.ndarray] = []
    risks: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if config.max_eval_batches and idx >= config.max_eval_batches:
                break
            batch = batch.to(device)
            nll, samples, crps_tensor = _distribution_metrics(model, batch, config.nsamples)
            mean = samples.mean(dim=1)
            mae, rmse = point_metrics(batch.y_flat, mean, batch.mq_flat)
            picp, mpiw = interval_metrics(batch.y_flat, samples, batch.mq_flat)
            crps = float(crps_tensor.cpu())
            samples_4d = samples.reshape(samples.shape[0], config.nsamples, config.pred_len, num_sensors)
            risk = threshold_risk_score(
                samples_4d,
                risk_upper_limits,
                risk_lower_limits,
            )
            label = _risk_labels(config.dataset, batch, config.risk_threshold)
            totals["nll"] += float(nll.cpu())
            totals["mae"] += mae
            totals["rmse"] += rmse
            totals["crps"] += crps
            totals["picp"] += picp
            totals["mpiw"] += mpiw
            means.append(mean.cpu().numpy())
            samples_out.append(samples.cpu().numpy())
            risks.append(risk.cpu().numpy())
            labels.append(label.cpu().numpy())
            count += 1
    elapsed = time.perf_counter() - start
    averaged = {key: value / max(count, 1) for key, value in totals.items()}
    risk_scores = np.concatenate(risks) if risks else np.array([])
    risk_labels = np.concatenate(labels) if labels else np.array([])
    averaged.update(safe_binary_metrics(risk_labels, risk_scores))
    averaged["ece"] = expected_calibration_error(risk_labels, risk_scores)
    averaged["lead_time"] = None
    averaged["infer_time_ms_per_batch"] = (elapsed / max(count, 1)) * 1000.0
    return averaged, means, samples_out, risks


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    spec = get_model_spec(config.model)
    if spec.status != "enabled":
        raise NotImplementedError(f"Model {config.model} is registered as {spec.status}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    output_dir = Path(config.output_dir)
    bundle = create_protocol_datasets(
        config.dataset,
        config.data_root,
        seed=config.seed,
        history_len=config.history_len,
        pred_len=config.pred_len,
        stride=config.stride,
        async_mode="none",
    )
    split_path = output_dir / "splits" / f"{config.dataset}_split_seed{config.seed}.json"
    _write_json(split_path, bundle.split_info)
    mask_path = output_dir / "masks" / f"{config.dataset}_missing_{config.missing_rate}_seed{config.seed}.npz"
    split_masks = generate_or_load_split_masks(
        mask_path,
        {
            "train": (len(bundle.train), config.history_len, bundle.num_sensors),
            "valid": (len(bundle.valid), config.history_len, bundle.num_sensors),
            "test": (len(bundle.test), config.history_len, bundle.num_sensors),
        },
        missing_rate=config.missing_rate,
        seed=config.seed,
        mode=config.missing_mode,
    )
    train_set = MaskedWindowDataset(bundle.train, split_masks["train"])
    valid_set = MaskedWindowDataset(bundle.valid, split_masks["valid"])
    test_set = MaskedWindowDataset(bundle.test, split_masks["test"])
    collator = IndustrialCollator()
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=collator)
    device = torch.device(config.device)
    risk_lower_limits = bundle.risk_lower_limits.to(device)
    risk_upper_limits = bundle.risk_upper_limits.to(device)
    model = create_model(
        config.model,
        num_sensors=bundle.num_sensors,
        context_dim=bundle.context_dim,
        device=config.device,
        hidden_dim=config.hidden_dim,
        te_dim=config.te_dim,
        kernel_count=config.kernel_count,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        flow_layers=config.flow_layers,
        preconv_dim=config.preconv_dim,
        lambda_point=config.lambda_point,
    )
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_start = time.perf_counter()
    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device, config.max_train_batches)
        valid_metrics, _, _, _ = _evaluate(
            model,
            valid_loader,
            device,
            config,
            bundle.num_sensors,
            risk_lower_limits,
            risk_upper_limits,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_nll": valid_metrics["nll"]})
    train_time = time.perf_counter() - train_start
    eval_metrics, means, samples, risks = _evaluate(
        model,
        test_loader,
        device,
        config,
        bundle.num_sensors,
        risk_lower_limits,
        risk_upper_limits,
    )
    pred_dir = output_dir / "predictions" / config.dataset / config.model
    pred_dir.mkdir(parents=True, exist_ok=True)
    mean_array = np.concatenate(means) if means else np.empty((0,))
    sample_array = np.concatenate(samples) if samples else np.empty((0,))
    risk_array = np.concatenate(risks) if risks else np.empty((0,))
    np.save(pred_dir / f"mean_seed{config.seed}.npy", mean_array)
    np.save(pred_dir / f"samples_seed{config.seed}.npy", sample_array)
    np.save(pred_dir / f"risk_seed{config.seed}.npy", risk_array)
    checkpoint_dir = output_dir / "checkpoints" / config.dataset / config.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model.config.to_dict(),
            "experiment": asdict(config),
            "history": history,
        },
        checkpoint_dir / f"checkpoint_seed{config.seed}.pt",
    )
    _write_config_yaml(output_dir / "configs" / f"{config.dataset}_{config.model}_seed{config.seed}.yaml", config)
    metrics = {
        "dataset": config.dataset,
        "model": config.model,
        "seed": config.seed,
        "missing_rate": config.missing_rate,
        "history": config.history_len,
        "horizon": config.pred_len,
        **completed_metrics_template(),
        **eval_metrics,
        "train_time_sec": train_time,
        "num_params": sum(param.numel() for param in model.parameters()),
        "gpu_memory_mb": None,
        "status": "completed",
        "error": None,
    }
    metrics_path = output_dir / "metrics" / config.dataset / config.model / f"metrics_seed{config.seed}.json"
    _write_json(metrics_path, metrics)
    return metrics


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run unified KAF-ProFITi experiment")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="kaf_profiti_joint")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--history-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--output-dir", default="code/results")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--max-eval-batches", type=int, default=20)
    parser.add_argument("--nsamples", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--missing-mode", default="mixed")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--risk-threshold", type=float, default=30.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--te-dim", type=int, default=5)
    parser.add_argument("--kernel-count", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--flow-layers", type=int, default=2)
    parser.add_argument("--preconv-dim", type=int, default=8)
    parser.add_argument("--lambda-point", type=float, default=0.1)
    return ExperimentConfig(**vars(parser.parse_args()))


def main():
    metrics = run_experiment(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

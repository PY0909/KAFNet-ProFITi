import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .registry import list_model_specs


SUMMARY_METRICS = [
    "mae",
    "rmse",
    "nll",
    "crps",
    "picp",
    "mpiw",
    "auroc",
    "auprc",
    "f1",
    "ece",
    "label_positive_rate",
    "risk_score_mean",
    "risk_score_std",
    "train_time_sec",
    "infer_time_ms_per_batch",
    "num_params",
    "gpu_memory_mb",
]


def _read_metrics(results_dir: Path) -> List[Dict[str, object]]:
    rows = []
    for path in sorted((results_dir / "metrics").glob("*/*/metrics_seed*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _mean_std_rows(metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not metrics:
        return []
    frame = pd.DataFrame(metrics)
    group_columns = [
        column
        for column in ["dataset", "model", "missing_rate", "history", "horizon"]
        if column in frame.columns
    ]
    rows = []
    for key, group in frame.groupby(group_columns, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, values))
        row["seed_count"] = int(group["seed"].nunique()) if "seed" in group else len(group)
        row["seeds"] = ",".join(str(int(seed)) for seed in sorted(group["seed"].dropna().unique()))
        row["status"] = "completed" if (group.get("status") == "completed").all() else "mixed"
        row["p_value"] = None
        for metric in SUMMARY_METRICS:
            if metric not in group:
                continue
            numeric = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(numeric.mean()) if not numeric.empty else None
            row[f"{metric}_std"] = float(numeric.std(ddof=1)) if len(numeric) > 1 else None
        rows.append(row)
    return rows


def build_tables(results_dir) -> None:
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    metrics = _read_metrics(results_dir)
    datasets = sorted({row["dataset"] for row in metrics})
    metric_by_key = {(row["dataset"], row["model"], row["seed"]): row for row in metrics}
    specs = list_model_specs()

    split_rows = []
    for split_path in sorted((results_dir / "splits").glob("*_split_seed*.json")):
        split = json.loads(split_path.read_text())
        split_rows.append(
            {
                "dataset": split.get("dataset"),
                "seed": split.get("seed"),
                "train_windows": split.get("train_windows"),
                "valid_windows": split.get("valid_windows"),
                "test_windows": split.get("test_windows"),
                "num_sensors": split.get("num_sensors"),
                "split_rule": split.get("split_rule"),
            }
        )
    _write_csv(tables_dir / "table1_dataset_split.csv", split_rows)

    table2 = []
    for dataset in datasets:
        seeds = sorted({row["seed"] for row in metrics if row["dataset"] == dataset})
        for spec in specs:
            for seed in seeds or [None]:
                row = metric_by_key.get((dataset, spec.name, seed))
                if row is None:
                    table2.append(
                        {
                            "dataset": dataset,
                            "model": spec.name,
                            "seed": seed,
                            "mae": None,
                            "rmse": None,
                            "nll": None,
                            "crps": None,
                            "status": spec.status,
                        }
                    )
                else:
                    table2.append(
                        {
                            "dataset": dataset,
                            "model": spec.name,
                            "seed": seed,
                            "mae": row.get("mae"),
                            "rmse": row.get("rmse"),
                            "nll": row.get("nll"),
                            "crps": row.get("crps"),
                            "status": row.get("status"),
                        }
                    )
    _write_csv(tables_dir / "table2_main_forecasting.csv", table2)

    _write_csv(
        tables_dir / "table3_risk_prediction.csv",
        [
            {
                "dataset": row.get("dataset"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "auroc": row.get("auroc"),
                "auprc": row.get("auprc"),
                "f1": row.get("f1"),
                "ece": row.get("ece"),
                "lead_time": row.get("lead_time"),
                "risk_label_mode": row.get("risk_label_mode"),
                "risk_score_rule": row.get("risk_score_rule"),
                "label_positive_rate": row.get("label_positive_rate"),
                "label_unique_count": row.get("label_unique_count"),
                "risk_score_min": row.get("risk_score_min"),
                "risk_score_max": row.get("risk_score_max"),
                "risk_score_mean": row.get("risk_score_mean"),
                "risk_score_std": row.get("risk_score_std"),
                "risk_score_is_constant": row.get("risk_score_is_constant"),
                "status": row.get("status"),
            }
            for row in metrics
        ],
    )
    _write_csv(
        tables_dir / "table4_missing_robustness.csv",
        [
            {
                "dataset": row.get("dataset"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "missing_rate": row.get("missing_rate"),
                "mae": row.get("mae"),
                "nll": row.get("nll"),
                "auroc": row.get("auroc"),
                "status": row.get("status"),
            }
            for row in metrics
        ],
    )
    ablation_models = {
        "kafnet",
        "kafnet_gaussian",
        "kaf_profiti_marginal",
        "kaf_profiti_joint_no_context",
        "kaf_profiti_joint",
    }
    _write_csv(
        tables_dir / "table5_ablation.csv",
        [row for row in table2 if row.get("model") in ablation_models],
    )
    _write_csv(
        tables_dir / "table6_efficiency.csv",
        [
            {
                "dataset": row.get("dataset"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "num_params": row.get("num_params"),
                "train_time_sec": row.get("train_time_sec"),
                "infer_time_ms_per_batch": row.get("infer_time_ms_per_batch"),
                "gpu_memory_mb": row.get("gpu_memory_mb"),
                "status": row.get("status"),
            }
            for row in metrics
        ],
    )
    _write_csv(
        tables_dir / "table7_statistical_test.csv",
        _mean_std_rows(metrics),
    )

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kaf_profiti.experiments.datasets import create_protocol_datasets, resolve_data_root
from kaf_profiti.experiments.masks import (
    MaskedWindowDataset,
    generate_or_load_masks,
    generate_or_load_split_masks,
)
from kaf_profiti.experiments.registry import create_model, get_model_spec, list_model_specs
from kaf_profiti.experiments.tables import build_tables
from run_experiment import ExperimentConfig, run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "dataset"


def _require_dataset(dataset_dir: Path):
    if not dataset_dir.exists():
        pytest.skip(f"本地小规模数据不存在，跳过需要真实数据的实验框架测试: {dataset_dir}")


def test_resolve_data_root_falls_back_to_project_root_when_running_from_code_dir(monkeypatch):
    _require_dataset(DATA_ROOT / "CMAPSSData")
    monkeypatch.chdir(PROJECT_ROOT / "code")

    assert resolve_data_root("dataset") == DATA_ROOT


def test_cmapss_protocol_reports_required_files_when_data_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="C-MAPSS 数据文件缺失"):
        create_protocol_datasets(
            "cmapss_fd001",
            tmp_path,
            seed=2026,
            history_len=12,
            pred_len=3,
            stride=120,
            async_mode="none",
        )


def test_registry_enables_only_kaf_profiti_joint():
    specs = {spec.name: spec for spec in list_model_specs()}

    assert specs["kaf_profiti_joint"].status == "enabled"
    assert specs["tcn_gaussian"].status == "not_implemented"
    assert get_model_spec("kaf_profiti_joint").display_name == "KAFNet + ProFITi Joint Flow"
    with pytest.raises(NotImplementedError):
        create_model("tcn_gaussian", num_sensors=15, context_dim=3, device="cpu")


def test_protocol_splits_match_metropt_and_cmapss_rules():
    _require_dataset(DATA_ROOT / "metropt+3+dataset")
    _require_dataset(DATA_ROOT / "CMAPSSData")
    metro = create_protocol_datasets(
        "metropt3",
        DATA_ROOT,
        seed=2026,
        history_len=12,
        pred_len=3,
        stride=50_000,
        async_mode="none",
    )
    cmapss = create_protocol_datasets(
        "cmapss_fd001",
        DATA_ROOT,
        seed=2026,
        history_len=30,
        pred_len=5,
        stride=50,
        async_mode="none",
    )

    assert metro.split_info["split_rule"] == "first_month_80_20_then_remaining_months"
    assert metro.split_info["train_end"] <= "2020-02-24T00:00:00"
    assert metro.split_info["test_start"] == "2020-03-01T00:00:00"
    assert len(metro.train) > 0 and len(metro.valid) > 0 and len(metro.test) > 0
    train_units = set(cmapss.split_info["train_engine_ids"])
    valid_units = set(cmapss.split_info["valid_engine_ids"])
    assert cmapss.split_info["split_rule"] == "engine_id_80_20_official_test"
    assert train_units
    assert valid_units
    assert train_units.isdisjoint(valid_units)


def test_metropt_protocol_normalization_does_not_explode_constant_train_channels():
    _require_dataset(DATA_ROOT / "metropt+3+dataset")
    metro = create_protocol_datasets(
        "metropt3",
        DATA_ROOT,
        seed=2026,
        history_len=12,
        pred_len=3,
        stride=50_000,
        async_mode="none",
    )

    sample = metro.test[0]

    assert torch.isfinite(sample.X_obs).all()
    assert torch.isfinite(sample.Y_q).all()
    assert sample.X_obs.abs().max() < 100.0
    assert sample.Y_q.abs().max() < 100.0


def test_mask_npz_roundtrip_and_dataset_wrapper(tmp_path):
    _require_dataset(DATA_ROOT / "CMAPSSData")
    bundle = create_protocol_datasets(
        "cmapss_fd001",
        DATA_ROOT,
        seed=2026,
        history_len=12,
        pred_len=3,
        stride=120,
        async_mode="none",
    )
    mask_path = tmp_path / "cmapss_fd001_missing_0.3_seed2026.npz"

    masks_a = generate_or_load_masks(
        mask_path,
        num_windows=len(bundle.train),
        history_len=12,
        num_sensors=21,
        missing_rate=0.3,
        seed=2026,
        mode="mixed",
    )
    masks_b = generate_or_load_masks(
        mask_path,
        num_windows=len(bundle.train),
        history_len=12,
        num_sensors=21,
        missing_rate=0.3,
        seed=2026,
        mode="mixed",
    )
    wrapped = MaskedWindowDataset(bundle.train, masks_a)
    sample = wrapped[0]

    assert mask_path.exists()
    assert masks_a.dtype == np.uint8
    assert np.array_equal(masks_a, masks_b)
    assert torch.equal(sample.M_obs, torch.tensor(masks_a[0], dtype=torch.float32))
    assert torch.all(sample.X_obs[sample.M_obs == 0] == 0)


def test_split_mask_regenerates_empty_cache_file(tmp_path):
    mask_path = tmp_path / "cmapss_fd001_missing_0.3_seed2028.npz"
    mask_path.touch()

    masks = generate_or_load_split_masks(
        mask_path,
        {
            "train": (2, 4, 3),
            "valid": (1, 4, 3),
            "test": (1, 4, 3),
        },
        missing_rate=0.3,
        seed=2028,
        mode="mixed",
    )

    assert mask_path.stat().st_size > 0
    assert masks["train"].shape == (2, 4, 3)
    assert masks["train"].dtype == np.uint8
    loaded = np.load(mask_path)
    assert loaded["valid"].shape == (1, 4, 3)


def test_mask_regenerates_cache_with_wrong_shape(tmp_path):
    mask_path = tmp_path / "mask.npz"
    np.savez_compressed(mask_path, mask=np.ones((1, 2, 3), dtype=np.uint8))

    masks = generate_or_load_masks(
        mask_path,
        num_windows=2,
        history_len=4,
        num_sensors=3,
        missing_rate=0.3,
        seed=2026,
        mode="mixed",
    )

    assert masks.shape == (2, 4, 3)
    assert np.load(mask_path)["mask"].shape == (2, 4, 3)


def test_run_experiment_writes_unified_outputs(tmp_path):
    _require_dataset(DATA_ROOT / "CMAPSSData")
    config = ExperimentConfig(
        dataset="cmapss_fd001",
        model="kaf_profiti_joint",
        seed=2026,
        missing_rate=0.3,
        history_len=12,
        pred_len=3,
        stride=120,
        data_root=str(DATA_ROOT),
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=2,
        max_train_batches=1,
        max_eval_batches=1,
        nsamples=3,
        device="cpu",
    )

    metrics = run_experiment(config)
    metrics_path = (
        tmp_path / "metrics" / "cmapss_fd001" / "kaf_profiti_joint" / "metrics_seed2026.json"
    )
    samples_path = (
        tmp_path / "predictions" / "cmapss_fd001" / "kaf_profiti_joint" / "samples_seed2026.npy"
    )

    assert metrics["status"] == "completed"
    assert metrics_path.exists()
    assert samples_path.exists()
    loaded = json.loads(metrics_path.read_text())
    assert loaded["dataset"] == "cmapss_fd001"
    assert loaded["model"] == "kaf_profiti_joint"
    assert loaded["missing_rate"] == 0.3
    assert loaded["mae"] is not None
    assert loaded["rmse"] is not None
    assert loaded["picp"] is not None


def test_build_tables_reads_metrics_and_marks_not_implemented(tmp_path):
    metrics_dir = tmp_path / "metrics" / "metropt3" / "kaf_profiti_joint"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics_seed2026.json").write_text(
        json.dumps(
            {
                "dataset": "metropt3",
                "model": "kaf_profiti_joint",
                "seed": 2026,
                "missing_rate": 0.3,
                "history": 12,
                "horizon": 3,
                "mae": 1.0,
                "rmse": 2.0,
                "nll": 3.0,
                "crps": 4.0,
                "picp": 0.9,
                "mpiw": 1.2,
                "auroc": None,
                "auprc": None,
                "f1": None,
                "ece": None,
                "lead_time": None,
                "train_time_sec": 1.0,
                "infer_time_ms_per_batch": 2.0,
                "num_params": 10,
                "gpu_memory_mb": None,
                "status": "completed",
                "error": None,
            }
        )
    )

    build_tables(tmp_path)

    table2 = tmp_path / "tables" / "table2_main_forecasting.csv"
    table5 = tmp_path / "tables" / "table5_ablation.csv"
    assert table2.exists()
    assert table5.exists()
    text = table2.read_text()
    assert "kaf_profiti_joint" in text
    assert "not_implemented" in text


def test_build_tables_adds_mean_std_summary(tmp_path):
    for seed, mae, rmse, nll in [
        (2026, 1.0, 2.0, 3.0),
        (2027, 3.0, 4.0, 5.0),
    ]:
        metrics_dir = tmp_path / "metrics" / "cmapss_fd001" / "kaf_profiti_joint"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"metrics_seed{seed}.json").write_text(
            json.dumps(
                {
                    "dataset": "cmapss_fd001",
                    "model": "kaf_profiti_joint",
                    "seed": seed,
                    "missing_rate": 0.3,
                    "history": 12,
                    "horizon": 3,
                    "mae": mae,
                    "rmse": rmse,
                    "nll": nll,
                    "crps": 0.5,
                    "picp": 0.9,
                    "mpiw": 1.2,
                    "auroc": 0.7,
                    "auprc": 0.8,
                    "f1": 0.6,
                    "ece": 0.1,
                    "lead_time": None,
                    "train_time_sec": 1.0,
                    "infer_time_ms_per_batch": 2.0,
                    "num_params": 10,
                    "gpu_memory_mb": None,
                    "status": "completed",
                    "error": None,
                }
            )
        )

    build_tables(tmp_path)

    table7 = tmp_path / "tables" / "table7_statistical_test.csv"
    text = table7.read_text()
    assert "mae_mean" in text
    assert "mae_std" in text
    assert "2.0" in text
    assert "1.4142135623730951" in text

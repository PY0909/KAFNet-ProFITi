import json
from pathlib import Path

from run_experiment_matrix import (
    Job,
    build_jobs,
    command_for_job,
    default_datasets_for_profile,
    default_models_for_group,
    should_skip_job,
)


def test_ablation_defaults_cover_main_datasets_models_and_three_seeds():
    datasets = default_datasets_for_profile("ablation")
    models = default_models_for_group("ablation")
    jobs = build_jobs(
        datasets=datasets,
        models=models,
        seeds=[2026, 2027, 2028],
        missing_rates=[0.3],
        history_len=96,
        pred_len=24,
        cmapss_stride=1,
        metropt_stride=120,
        tep_stride=250,
    )

    assert datasets == ["cmapss_fd001", "metropt3"]
    assert models == [
        "kafnet_gaussian",
        "kaf_profiti_marginal",
        "kaf_profiti_joint_no_context",
        "kaf_profiti_joint",
    ]
    assert len(jobs) == 24
    assert {job.stride for job in jobs if job.dataset == "cmapss_fd001"} == {1}
    assert {job.stride for job in jobs if job.dataset == "metropt3"} == {120}


def test_all_model_group_includes_real_baselines_and_ablation_models():
    models = default_models_for_group("all")

    assert models == [
        "tcn_gaussian",
        "gru_d",
        "kafnet_gaussian",
        "kaf_profiti_marginal",
        "kaf_profiti_joint_no_context",
        "kaf_profiti_joint",
    ]


def test_command_for_job_uses_full_run_limits_and_dataset_arguments():
    job = Job(
        dataset="cmapss_fd001",
        model="kaf_profiti_joint",
        seed=2026,
        missing_rate=0.3,
        stride=1,
    )

    command = command_for_job(
        job,
        python_executable="/env/bin/python",
        run_script=Path("code/run_experiment.py"),
        data_root="dataset",
        output_dir="code/results",
        history_len=96,
        pred_len=24,
        epochs=1,
        batch_size=16,
        max_train_batches=0,
        max_eval_batches=0,
        nsamples=20,
        device="cuda",
        risk_label_mode="pre_fault_6h",
    )

    assert command[:2] == ["/env/bin/python", "code/run_experiment.py"]
    assert command[command.index("--dataset") + 1] == "cmapss_fd001"
    assert command[command.index("--model") + 1] == "kaf_profiti_joint"
    assert command[command.index("--max-train-batches") + 1] == "0"
    assert command[command.index("--max-eval-batches") + 1] == "0"
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--risk-label-mode") + 1] == "pre_fault_6h"


def test_should_skip_job_when_completed_metrics_exists(tmp_path):
    job = Job(
        dataset="cmapss_fd001",
        model="kaf_profiti_joint",
        seed=2026,
        missing_rate=0.3,
        stride=1,
    )
    metrics_path = (
        tmp_path
        / "metrics"
        / "cmapss_fd001"
        / "kaf_profiti_joint"
        / "metrics_seed2026.json"
    )
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(json.dumps({"status": "completed"}))

    assert should_skip_job(job, tmp_path)

#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

ENABLED_ABLATION_MODELS = [
    "kafnet_gaussian",
    "kaf_profiti_marginal",
    "kaf_profiti_joint_no_context",
    "kaf_profiti_joint",
]
ENABLED_BASELINE_MODELS = [
    "tcn_gaussian",
    "gru_d",
]
MAIN_DATASETS = ["cmapss_fd001", "metropt3"]
CMAPSS_ALL_DATASETS = ["cmapss_fd001", "cmapss_fd002", "cmapss_fd003", "cmapss_fd004"]


@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    seed: int
    missing_rate: float
    stride: int


def _split_csv(value: str, cast=str) -> List[object]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def default_models_for_group(group: str) -> List[str]:
    if group == "ablation":
        return list(ENABLED_ABLATION_MODELS)
    if group == "all":
        return list(ENABLED_BASELINE_MODELS) + list(ENABLED_ABLATION_MODELS)
    if group == "final":
        return ["kaf_profiti_joint"]
    raise ValueError(f"Unknown model group: {group}")


def default_datasets_for_profile(profile: str) -> List[str]:
    if profile in {"ablation", "main"}:
        return list(MAIN_DATASETS)
    if profile == "cmapss_all":
        return list(CMAPSS_ALL_DATASETS)
    if profile == "cmapss_fd001":
        return ["cmapss_fd001"]
    if profile == "metropt3":
        return ["metropt3"]
    raise ValueError(f"Unknown experiment profile: {profile}")


def stride_for_dataset(dataset: str, cmapss_stride: int, metropt_stride: int, tep_stride: int) -> int:
    if dataset.startswith("cmapss"):
        return int(cmapss_stride)
    if dataset == "metropt3":
        return int(metropt_stride)
    if dataset == "tep":
        return int(tep_stride)
    raise ValueError(f"Unknown dataset: {dataset}")


def build_jobs(
    datasets: Sequence[str],
    models: Sequence[str],
    seeds: Sequence[int],
    missing_rates: Sequence[float],
    history_len: int,
    pred_len: int,
    cmapss_stride: int,
    metropt_stride: int,
    tep_stride: int,
) -> List[Job]:
    del history_len, pred_len
    jobs = []
    for dataset in datasets:
        stride = stride_for_dataset(dataset, cmapss_stride, metropt_stride, tep_stride)
        for model in models:
            for seed in seeds:
                for missing_rate in missing_rates:
                    jobs.append(Job(dataset, model, int(seed), float(missing_rate), stride))
    return jobs


def metrics_path_for_job(job: Job, output_dir: Path) -> Path:
    return output_dir / "metrics" / job.dataset / job.model / f"metrics_seed{job.seed}.json"


def should_skip_job(job: Job, output_dir: Path) -> bool:
    path = metrics_path_for_job(job, output_dir)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed"


def command_for_job(
    job: Job,
    python_executable: str,
    run_script: Path,
    data_root: str,
    output_dir: str,
    history_len: int,
    pred_len: int,
    epochs: int,
    batch_size: int,
    max_train_batches: int,
    max_eval_batches: int,
    nsamples: int,
    device: str,
    risk_label_mode: str,
) -> List[str]:
    return [
        python_executable,
        str(run_script),
        "--dataset",
        job.dataset,
        "--model",
        job.model,
        "--seed",
        str(job.seed),
        "--missing-rate",
        str(job.missing_rate),
        "--history-len",
        str(history_len),
        "--pred-len",
        str(pred_len),
        "--stride",
        str(job.stride),
        "--data-root",
        data_root,
        "--output-dir",
        output_dir,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--max-train-batches",
        str(max_train_batches),
        "--max-eval-batches",
        str(max_eval_batches),
        "--nsamples",
        str(nsamples),
        "--device",
        device,
        "--risk-label-mode",
        risk_label_mode,
    ]


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run KAF-ProFITi experiment matrix")
    parser.add_argument("--profile", default="ablation", choices=["ablation", "main", "cmapss_all", "cmapss_fd001", "metropt3"])
    parser.add_argument("--model-group", default="ablation", choices=["ablation", "final", "all"])
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset override")
    parser.add_argument("--models", default=None, help="Comma-separated model override")
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--missing-rates", default="0.3")
    parser.add_argument("--history-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--cmapss-stride", type=int, default=1)
    parser.add_argument("--metropt-stride", type=int, default=120)
    parser.add_argument("--tep-stride", type=int, default=250)
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--output-dir", default="code/results")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--risk-label-mode",
        default="pre_fault_6h",
        choices=["fault_window", "pre_fault_1h", "pre_fault_6h", "pre_fault_24h"],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-completed", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-build-tables", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = _split_csv(args.datasets) if args.datasets else default_datasets_for_profile(args.profile)
    models = _split_csv(args.models) if args.models else default_models_for_group(args.model_group)
    seeds = _split_csv(args.seeds, int)
    missing_rates = _split_csv(args.missing_rates, float)
    output_dir = Path(args.output_dir)
    run_script = Path(__file__).resolve().with_name("run_experiment.py")
    jobs = build_jobs(
        datasets=datasets,
        models=models,
        seeds=seeds,
        missing_rates=missing_rates,
        history_len=args.history_len,
        pred_len=args.pred_len,
        cmapss_stride=args.cmapss_stride,
        metropt_stride=args.metropt_stride,
        tep_stride=args.tep_stride,
    )
    log_path = output_dir / "logs" / "experiment_matrix.jsonl"
    print(f"计划任务数: {len(jobs)}")
    print(f"数据集: {', '.join(datasets)}")
    print(f"模型: {', '.join(models)}")
    print(f"seeds: {', '.join(str(seed) for seed in seeds)}")
    failures = []
    completed = 0
    skipped = 0
    env = os.environ.copy()
    env["PYTHONPATH"] = "code" if not env.get("PYTHONPATH") else f"code{os.pathsep}{env['PYTHONPATH']}"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    for index, job in enumerate(jobs, start=1):
        command = command_for_job(
            job,
            python_executable=args.python,
            run_script=run_script,
            data_root=args.data_root,
            output_dir=args.output_dir,
            history_len=args.history_len,
            pred_len=args.pred_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            nsamples=args.nsamples,
            device=args.device,
            risk_label_mode=args.risk_label_mode,
        )
        if not args.no_skip_completed and should_skip_job(job, output_dir):
            skipped += 1
            print(f"[{index}/{len(jobs)}] 跳过已完成: {job.dataset} {job.model} seed={job.seed}")
            continue
        print(f"[{index}/{len(jobs)}] 运行: {' '.join(command)}")
        if args.dry_run:
            continue
        start = subprocess.run(command, env=env)
        record = {
            "dataset": job.dataset,
            "model": job.model,
            "seed": job.seed,
            "missing_rate": job.missing_rate,
            "returncode": start.returncode,
        }
        _write_jsonl(log_path, [record])
        if start.returncode == 0:
            completed += 1
        else:
            failures.append(record)
            if args.stop_on_error:
                break
    if not args.dry_run and not args.no_build_tables:
        table_command = [
            args.python,
            str(Path(__file__).resolve().with_name("build_tables.py")),
            "--results-dir",
            args.output_dir,
        ]
        subprocess.run(table_command, env=env, check=True)
    print(f"完成: {completed}, 跳过: {skipped}, 失败: {len(failures)}")
    if failures:
        print(json.dumps(failures, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

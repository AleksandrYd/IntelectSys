import argparse
import math
import time
import traceback
import uuid
from importlib.metadata import version
from pathlib import Path

import mlflow
import optuna
import torch
import yaml
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from harness import (
    ROOT,
    append_history,
    commit_hash,
    file_sha256,
    now,
    recompute_metrics,
    setup_mlflow,
    write_json,
)
from model import MLP
from train import evaluate, load_arrays, prepare_seed, write_predictions


def trial_config(params: dict) -> dict:
    return {
        "experiment": "lr0",
        "split_seed": 0,
        "seed": 0,
        "model": {
            "n_layers": int(params["n_layers"]),
            "hidden": int(params["hidden"]),
            "dropout": float(params["dropout"]),
        },
        "train": {
            "lr": float(params["lr"]),
            "weight_decay": float(params["weight_decay"]),
            "batch_size": int(params["batch_size"]),
            "epochs": 30,
        },
        "runtime": {"device": "cpu", "num_threads": 1},
        "eval_test": False,
    }


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "hidden": trial.suggest_int("hidden", 32, 512, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def train_trial(cfg: dict, x, y, split, run_dir: Path, trial: optuna.Trial) -> float:
    prepare_seed(cfg["seed"], cfg["runtime"]["num_threads"])
    device = torch.device(cfg["runtime"]["device"])
    model = MLP(**cfg["model"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    x_train = torch.tensor(x[split["train"]], device=device)
    y_train = torch.tensor(y[split["train"]], device=device)
    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    x_val = torch.tensor(x[split["val"]], device=device)
    y_val = torch.tensor(y[split["val"]], device=device)
    val_acc = float("nan")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        seen = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            seen += len(yb)
        _, val_acc, _ = evaluate(model, x_val, y_val, criterion)
        row = {
            "epoch": epoch,
            "train_loss": f"{total_loss / seen:.17g}",
            "val_loss": f"{evaluate(model, x_val, y_val, criterion)[0]:.17g}",
            "val_acc": f"{val_acc:.17g}",
            "epoch_time_s": f"{(time.perf_counter() - started):.17g}",
        }
        append_history(run_dir / "history.csv", row)
        mlflow.log_metric("train_loss", float(row["train_loss"]), step=epoch)
        mlflow.log_metric("val_loss", float(row["val_loss"]), step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        trial.report(val_acc, step=epoch)
        if trial.should_prune():
            save_validation(model, x, y, split, device, criterion, run_dir)
            raise optuna.TrialPruned()

    save_validation(model, x, y, split, device, criterion, run_dir)
    return val_acc


def save_validation(model, x, y, split, device, criterion, run_dir: Path) -> None:
    indexes = split["val"]
    _, _, pred = evaluate(
        model,
        torch.tensor(x[indexes], device=device),
        torch.tensor(y[indexes], device=device),
        criterion,
    )
    rows = [
        {"split": "val", "index": int(index), "y_true": int(true_label), "y_pred": int(predicted)}
        for index, true_label, predicted in zip(indexes, y[indexes].tolist(), pred.tolist())
    ]
    write_predictions(run_dir / "predictions.csv", rows)


def objective(trial: optuna.Trial, x, y, split, commit: str) -> float:
    params = suggest_params(trial)
    cfg = trial_config(params)
    run_id = uuid.uuid4().hex
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    metadata = {
        "run_id": run_id,
        "series": "lr0_search",
        "trial_number": trial.number,
        "status": "running",
        "started_at": now(),
        "ended_at": None,
        "commit": commit,
        "debug": False,
        "versions": {
            "python": __import__("platform").python_version(),
            "torch": version("torch"),
            "scikit-learn": version("scikit-learn"),
            "optuna": version("optuna"),
            "mlflow": version("mlflow"),
        },
        "num_threads": 1,
        "seed": 0,
        "split_sha256": file_sha256(ROOT / "data" / "split.json"),
    }
    write_json(run_dir / "metadata.json", metadata)

    pruned = False
    val_acc = float("nan")
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.set_tags({"commit": commit, "series": "lr0_search", "seed": "0", "trial": str(trial.number)})
        try:
            val_acc = train_trial(cfg, x, y, split, run_dir, trial)
            state = "FAIL" if math.isnan(val_acc) else "COMPLETE"
        except optuna.TrialPruned:
            pruned = True
            state = "PRUNED"
        except Exception:
            metadata["status"] = "failed"
            metadata["ended_at"] = now()
            metadata["traceback"] = traceback.format_exc()
            write_json(run_dir / "metadata.json", metadata)
            raise
        mlflow.set_tag("state", state)
        metadata["status"] = state.lower()
        metadata["ended_at"] = now()
        write_json(run_dir / "metadata.json", metadata)
        if (run_dir / "predictions.csv").exists():
            metrics = recompute_metrics(run_dir)
            write_json(run_dir / "metrics.json", metrics)
            mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, float)})
        mlflow.log_artifacts(str(run_dir))
    if pruned:
        raise optuna.TrialPruned()
    return val_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-trials", type=int, required=True)
    args = parser.parse_args()

    commit = commit_hash()
    x, y, split = load_arrays({"split_seed": 0})
    setup_mlflow()
    study = optuna.create_study(
        study_name="lr0_search",
        storage="sqlite:///" + (ROOT / "optuna.db").resolve().as_posix(),
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    with mlflow.start_run(run_name="lr0_search"):
        mlflow.set_tags({"commit": commit, "series": "lr0_search"})
        study.optimize(
            lambda trial: objective(trial, x, y, split, commit),
            n_trials=args.max_trials,
            callbacks=[MaxTrialsCallback(args.max_trials, states=None)],
        )
    counts = {state.name: 0 for state in TrialState}
    for trial in study.trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
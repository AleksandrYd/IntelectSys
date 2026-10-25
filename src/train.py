import argparse
import traceback
import uuid
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.datasets import load_digits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import mlflow
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


def load_config(path: Path, seed: int | None) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if seed is not None:
        cfg["seed"] = seed
    lr = cfg["train"]["lr"]
    if lr <= 0:
        raise SystemExit(f"lr должен быть больше 0, получено {lr}")
    if cfg["train"]["epochs"] != 30:
        raise SystemExit("В этой работе число эпох фиксировано: 30")
    if not 0 <= cfg["model"]["dropout"] <= 0.5:
        raise SystemExit("dropout должен лежать от 0 до 0.5")
    if cfg["model"]["n_layers"] not in (1, 2, 3):
        raise SystemExit("n_layers должен быть 1, 2 или 3")
    return cfg


def prepare_seed(seed: int, num_threads: int) -> None:
    torch.set_num_threads(num_threads)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_arrays(cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    split_path = ROOT / "data" / "split.json"
    manifest = yaml.safe_load((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    # manifest.json — JSON, читаем его отдельным модулем
    import json

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["digits"]["files"]["data/split.json"]["sha256"]
    if file_sha256(split_path) != recorded:
        raise SystemExit("SHA-256 data/split.json не совпадает с манифестом")
    if cfg["split_seed"] != manifest["digits"]["split_seed"]:
        raise SystemExit("split_seed конфигурации не совпадает с манифестом")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    x, y = load_digits(return_X_y=True)
    return x.astype(np.float32) / 16.0, y.astype(np.int64), split


def fit(cfg: dict, x: np.ndarray, y: np.ndarray, split: dict, eval_test: bool, run_dir: Path) -> None:
    seed = cfg["seed"]
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
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    x_val = torch.tensor(x[split["val"]], device=device)
    y_val = torch.tensor(y[split["val"]], device=device)

    import time

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
        val_loss, val_acc, _ = evaluate(model, x_val, y_val, criterion)
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": f"{total_loss / seen:.17g}",
            "val_loss": f"{val_loss:.17g}",
            "val_acc": f"{val_acc:.17g}",
            "epoch_time_s": f"{elapsed:.17g}",
        }
        append_history(run_dir / "history.csv", row)
        mlflow.log_metric("train_loss", float(row["train_loss"]), step=epoch)
        mlflow.log_metric("val_loss", float(row["val_loss"]), step=epoch)
        mlflow.log_metric("val_acc", float(row["val_acc"]), step=epoch)

    parts = [("val", split["val"])]
    if eval_test:
        parts.append(("test", split["test"]))
    rows = []
    for name, indexes in parts:
        indexes = np.asarray(indexes)
        _, _, pred = evaluate(
            model,
            torch.tensor(x[indexes], device=device),
            torch.tensor(y[indexes], device=device),
            criterion,
        )
        for index, true_label, predicted in zip(indexes.tolist(), y[indexes].tolist(), pred.tolist()):
            rows.append(
                {"split": name, "index": index, "y_true": int(true_label), "y_pred": int(predicted)}
            )
    write_predictions(run_dir / "predictions.csv", rows)


def evaluate(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y).item()
        pred = logits.argmax(dim=1)
        accuracy = (pred == y).sum().item() / len(y)
    return loss, accuracy, pred.cpu().numpy()


def write_predictions(path: Path, rows: list[dict]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "index", "y_true", "y_pred"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--series", default="single")
    args = parser.parse_args()

    cfg = load_config(args.config, args.seed)
    cfg["eval_test"] = args.eval_test
    prepare_seed(cfg["seed"], cfg["runtime"]["num_threads"])
    x, y, split = load_arrays(cfg)
    commit = "debug" if args.debug else commit_hash()

    run_id = uuid.uuid4().hex
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    metadata = {
        "run_id": run_id,
        "series": args.series,
        "status": "running",
        "started_at": now(),
        "ended_at": None,
        "commit": commit,
        "debug": args.debug,
        "python": version("torch") and __import__("platform").python_version(),
        "versions": {
            "python": __import__("platform").python_version(),
            "torch": version("torch"),
            "scikit-learn": version("scikit-learn"),
            "optuna": version("optuna"),
            "mlflow": version("mlflow"),
        },
        "num_threads": cfg["runtime"]["num_threads"],
        "seed": cfg["seed"],
        "split_sha256": file_sha256(ROOT / "data" / "split.json"),
    }
    write_json(run_dir / "metadata.json", metadata)

    setup_mlflow()
    try:
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params(
                {
                    "seed": cfg["seed"],
                    "lr": cfg["train"]["lr"],
                    "weight_decay": cfg["train"]["weight_decay"],
                    "batch_size": cfg["train"]["batch_size"],
                    "epochs": cfg["train"]["epochs"],
                    "n_layers": cfg["model"]["n_layers"],
                    "hidden": cfg["model"]["hidden"],
                    "dropout": cfg["model"]["dropout"],
                    "eval_test": cfg["eval_test"],
                }
            )
            mlflow.set_tags(
                {
                    "commit": commit,
                    "series": args.series,
                    "seed": str(cfg["seed"]),
                    "debug": str(args.debug).lower(),
                }
            )
            fit(cfg, x, y, split, args.eval_test, run_dir)
            metrics = recompute_metrics(run_dir)
            write_json(run_dir / "metrics.json", metrics)
            metadata["status"] = "completed"
            metadata["ended_at"] = now()
            write_json(run_dir / "metadata.json", metadata)
            mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, float)})
            mlflow.log_artifacts(str(run_dir))
    except Exception:
        metadata["status"] = "failed"
        metadata["ended_at"] = now()
        metadata["traceback"] = traceback.format_exc()
        write_json(run_dir / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
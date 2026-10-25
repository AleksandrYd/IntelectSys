import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit_hash() -> str:
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "Есть незакоммиченные изменения — итоговый запуск невозможен:\n" + dirty
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_history(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "epoch_time_s"],
        )
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def recompute_metrics(run_dir: Path) -> dict:
    with (run_dir / "history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    with (run_dir / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if not history or not predictions:
        raise SystemExit("history.csv или predictions.csv пусты")

    def accuracy(split: str) -> tuple[float, int] | None:
        rows = [row for row in predictions if row["split"] == split]
        if not rows:
            return None
        correct = sum(int(row["y_true"]) == int(row["y_pred"]) for row in rows)
        return correct / len(rows), len(rows)

    val_acc, n_val = accuracy("val")
    metrics = {
        "val_acc": val_acc,
        "val_loss": float(history[-1]["val_loss"]),
        "train_time_s": sum(float(row["epoch_time_s"]) for row in history),
        "n_val": n_val,
    }
    test = accuracy("test")
    if test is not None:
        metrics["test_acc"], metrics["n_test"] = test
    return metrics


def setup_mlflow() -> None:
    db = (ROOT / "mlflow.db").resolve()
    mlflow.set_tracking_uri("sqlite:///" + db.as_posix())
    name = "lr0"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(name)
import csv
import json
import statistics
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]


def load_run(run_dir: Path) -> dict | None:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("series") != "final" or metadata.get("status") != "completed":
        return None
    config = (run_dir / "config.yaml").read_text(encoding="utf-8")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((run_dir / "predictions.csv").open(encoding="utf-8")))
    test = {
        int(row["index"]): (int(row["y_true"]), int(row["y_pred"]))
        for row in rows
        if row["split"] == "test"
    }
    return {
        "seed": metadata["seed"],
        "kind": "base" if "n_layers: 1" in config else "best",
        "val_acc": metrics["val_acc"],
        "test_acc": metrics["test_acc"],
        "test": test,
    }


runs = [item for item in (load_run(path) for path in (ROOT / "runs").iterdir()) if item]
by_seed = {}
for run in runs:
    by_seed.setdefault(run["seed"], {})[run["kind"]] = run

deltas = []
fixed = spoiled = 0
for seed in sorted(by_seed):
    base, best = by_seed[seed]["base"], by_seed[seed]["best"]
    deltas.append(
        {
            "seed": seed,
            "delta_val": best["val_acc"] - base["val_acc"],
            "delta_test": best["test_acc"] - base["test_acc"],
        }
    )
    for index, (true_label, base_pred) in base["test"].items():
        best_pred = best["test"][index][1]
        base_ok = base_pred == true_label
        best_ok = best_pred == true_label
        fixed += best_ok and not base_ok
        spoiled += base_ok and not best_ok

test_deltas = [row["delta_test"] for row in deltas]
val_deltas = [row["delta_val"] for row in deltas]
best_runs = [by_seed[seed]["best"] for seed in sorted(by_seed)]
study = optuna.load_study(
    study_name="lr0_search",
    storage="sqlite:///" + (ROOT / "optuna.db").resolve().as_posix(),
)
base_val = []
for run_dir in (ROOT / "runs").iterdir():
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("series") == "base_seeds" and metadata.get("status") == "completed":
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        base_val.append(metrics["val_acc"])

summary = {
    "deltas": deltas,
    "delta_val_median": statistics.median(val_deltas),
    "delta_val_min": min(val_deltas),
    "delta_val_max": max(val_deltas),
    "delta_test_median": statistics.median(test_deltas),
    "delta_test_min": min(test_deltas),
    "delta_test_max": max(test_deltas),
    "test_fixed": fixed,
    "test_spoiled": spoiled,
    "test_pairs": 5 * 360,
    "best_trial_val_acc": study.best_value,
    "best_new_seeds_val_mean": statistics.fmean(run["val_acc"] for run in best_runs),
    "best_new_seeds_test_mean": statistics.fmean(run["test_acc"] for run in best_runs),
    "base_seeds_val_spread": max(base_val) - min(base_val),
}
out = ROOT / "report" / "final_summary.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(out)
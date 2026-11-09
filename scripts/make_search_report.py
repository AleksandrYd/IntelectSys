import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import optuna

import mlflow

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)

study = optuna.load_study(
    study_name="lr0_search",
    storage="sqlite:///" + (ROOT / "optuna.db").resolve().as_posix(),
)
frame = study.trials_dataframe().sort_values("number")

counts = frame["state"].value_counts().to_dict()
epochs_by_trial = {}
epoch_times = []
for run_dir in (ROOT / "runs").iterdir():
    metadata_path = run_dir / "metadata.json"
    history_path = run_dir / "history.csv"
    if not metadata_path.exists() or not history_path.exists():
        continue
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("series") != "lr0_search":
        continue
    rows = list(csv.DictReader(history_path.open(encoding="utf-8")))
    epochs_by_trial[int(metadata["trial_number"])] = rows
    epoch_times.extend(float(row["epoch_time_s"]) for row in rows)

epochs_done = sum(len(rows) for rows in epochs_by_trial.values())
median_epoch_s = statistics.median(epoch_times)
time_saved_s = (1200 - epochs_done) * median_epoch_s

summary = {
    "n_trials": int(len(frame)),
    "counts": {str(state): int(count) for state, count in counts.items()},
    "epochs_done": epochs_done,
    "epochs_budget": 1200,
    "median_epoch_s": median_epoch_s,
    "time_saved_s": time_saved_s,
}
(OUT / "search_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

def last_val_acc(number: int):
    rows = epochs_by_trial.get(int(number))
    if not rows:
        return None
    return float(rows[-1]["val_acc"])

y = [last_val_acc(number) for number in frame["number"]]
best = []
current = None
for state, value in zip(frame["state"], frame["value"]):
    if state == "COMPLETE":
        current = value if current is None else max(current, value)
    best.append(current)

fig, ax = plt.subplots(figsize=(7, 4))
for state, marker in (("COMPLETE", "o"), ("PRUNED", "x"), ("FAIL", "s")):
    xs, ys = [], []
    for number, accuracy, row_state in zip(frame["number"], y, frame["state"]):
        if row_state == state and accuracy is not None:
            xs.append(number)
            ys.append(accuracy)
    ax.scatter(xs, ys, marker=marker, label=state)
ax.plot(frame["number"], best, label="лучшее на данный момент")
ax.set_xlabel("номер trial")
ax.set_ylabel("точность на val")
ax.legend()
fig.tight_layout()
history_path = OUT / "trial_history.png"
fig.savefig(history_path, dpi=150)
plt.close(fig)

importances = optuna.importance.get_param_importances(study)
fig, ax = plt.subplots(figsize=(7, 4))
names = list(importances)[::-1]
ax.barh(names, [importances[name] for name in names])
ax.set_xlabel("важность")
fig.tight_layout()
importance_path = OUT / "param_importance.png"
fig.savefig(importance_path, dpi=150)
plt.close(fig)

mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").resolve().as_posix())
mlflow.set_experiment("lr0")
with mlflow.start_run(run_name="search-report"):
    mlflow.log_artifact(str(OUT / "search_summary.json"))
    mlflow.log_artifact(str(history_path))
    mlflow.log_artifact(str(importance_path))

print(json.dumps(summary, ensure_ascii=False, indent=2))
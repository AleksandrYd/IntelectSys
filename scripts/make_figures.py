from pathlib import Path

import matplotlib.pyplot as plt
import optuna

import mlflow

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)

study = optuna.load_study(study_name="lr0_search", storage="sqlite:///optuna.db")
frame = study.trials_dataframe().sort_values("number")

best = []
current = None
for state, value in zip(frame["state"], frame["value"]):
    if state == "COMPLETE":
        current = value if current is None else max(current, value)
    best.append(current)

fig, ax = plt.subplots(figsize=(7, 4))
for state, marker in (("COMPLETE", "o"), ("PRUNED", "x"), ("FAIL", "s")):
    part = frame[frame["state"] == state]
    ax.scatter(part["number"], part["value"], marker=marker, label=state)
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
values = [importances[name] for name in names]
ax.barh(names, values)
ax.set_xlabel("важность")
fig.tight_layout()
importance_path = OUT / "param_importance.png"
fig.savefig(importance_path, dpi=150)
plt.close(fig)

mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").resolve().as_posix())
mlflow.set_experiment("lr0")
with mlflow.start_run(run_name="report-figures"):
    mlflow.log_artifact(str(history_path))
    mlflow.log_artifact(str(importance_path))

print(history_path)
print(importance_path)
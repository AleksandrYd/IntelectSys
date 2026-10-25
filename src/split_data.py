import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    X, y = load_digits(return_X_y=True)
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    train, val = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=0)

    parts = {"train": train, "val": val, "test": test}
    for name, part in parts.items():
        part.sort()
        labels = set(y[part].tolist())
        if labels != set(range(10)):
            raise SystemExit(f"В части {name} не все классы: {sorted(labels)}")

    merged = np.concatenate([train, val, test])
    if len(np.unique(merged)) != 1797 or len(merged) != 1797:
        raise SystemExit("Части пересекаются или не покрывают все 1797 изображений")
    if (len(train), len(val), len(test)) != (1077, 360, 360):
        raise SystemExit(f"Неверные размеры: {len(train)}, {len(val)}, {len(test)}")

    DATA.mkdir(exist_ok=True)
    split_path = DATA / "split.json"
    payload = {
        "train": train.tolist(),
        "val": val.tolist(),
        "test": test.tolist(),
    }
    split_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()

    manifest = {
        "digits": {
            "source": "scikit-learn",
            "version": version("scikit-learn"),
            "loader": "sklearn.datasets.load_digits",
            "n_total": 1797,
            "split_seed": 0,
            "rule": (
                "Сначала test_size=0.2 от всей выборки, затем val_size=0.25 "
                "от остатка. Оба вызова train_test_split стратифицированы по классу, "
                "random_state=0. Индексы в каждой части отсортированы."
            ),
            "n_train": 1077,
            "n_val": 360,
            "n_test": 360,
            "n_excluded": 0,
            "files": {
                "data/split.json": {"sha256": digest, "n_records": 1797}
            },
        }
    }
    (DATA / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"1077 / 360 / 360, sha256={digest}")


if __name__ == "__main__":
    main()
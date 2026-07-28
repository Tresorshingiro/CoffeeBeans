"""Register the shipped model as the reigning champion.

Without this the model_registry table is empty, so the promotion gate in
src/model.py compares candidates against 0.0 and would promote any
retrained model regardless of quality. Run once after setup, and at image
build time:

    python scripts/register_baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf  # noqa: E402

from src import config, database, model as model_module, preprocessing  # noqa: E402


def main() -> int:
    database.init_db()

    existing = database.get_champion()
    if existing:
        print(f"champion already registered: {existing['version']} "
              f"@ {existing['accuracy']:.4f} — nothing to do")
        return 0

    if not config.CHAMPION_PATH.exists():
        print(f"no model at {config.CHAMPION_PATH} — cannot register")
        return 1

    print(f"loading {config.CHAMPION_PATH}")
    model = tf.keras.models.load_model(config.CHAMPION_PATH)

    paths, labels = preprocessing.eval_slice()
    if not paths:
        print("no test images available — cannot score the baseline")
        return 1
    print(f"scoring against {len(paths)} held-out test images")

    ds = preprocessing.dataset_from_paths(paths, labels, shuffle=False)
    metrics = model_module.evaluate(model, ds)

    database.register_model("coffee_model", str(config.CHAMPION_PATH),
                            metrics["accuracy"])
    print(f"registered baseline 'coffee_model' at accuracy "
          f"{metrics['accuracy']:.4f} (loss {metrics['loss']:.4f})")
    for name, scores in metrics["per_class"].items():
        print(f"  {name:10s} precision={scores['precision']:.3f} "
              f"recall={scores['recall']:.3f} f1={scores['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Precompute the dataset visualizations served by /api/insights.

Recomputing these per page load on a 2-vCPU Space would be unusably slow.
Run once after the dataset is in place:  python scripts/build_insights.py
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

SAMPLE_PER_CLASS = 150


def _class_files(class_name):
    class_dir = config.train_dir() / class_name
    return sorted(p for p in class_dir.iterdir()
                  if p.suffix.lower() in config.ALLOWED_IMAGE_SUFFIXES)


def class_counts():
    return {c: len(_class_files(c)) for c in config.CLASS_NAMES}


BEAN_THRESHOLD = 200  # pixels darker than this are bean, not background


def channel_means(rng):
    """Mean RGB of the BEAN pixels only, excluding the white background.

    Averaging the whole frame measures the backdrop, not the bean: every
    class comes out near 210 and the differences vanish. Masking to the
    bean is what makes this chart say anything.
    """
    out = {}
    for class_name in config.CLASS_NAMES:
        files = _class_files(class_name)
        chosen = rng.sample(files, min(SAMPLE_PER_CLASS, len(files)))
        per_image = []
        for path in chosen:
            rgb = np.asarray(
                Image.open(path).convert("RGB").resize((128, 128)),
                dtype="float32")
            gray = np.asarray(
                Image.open(path).convert("L").resize((128, 128)),
                dtype="float32")
            mask = gray < BEAN_THRESHOLD
            if mask.sum():
                per_image.append(rgb[mask].mean(axis=0))
        means = np.mean(per_image, axis=0)
        out[class_name] = {"r": round(float(means[0]), 2),
                           "g": round(float(means[1]), 2),
                           "b": round(float(means[2]), 2),
                           "brightness": round(float(means.mean()), 2)}
    return out


def area_ratios(rng):
    """Fraction of pixels belonging to the bean rather than the background.

    Beans are photographed on a plain white background, so a simple
    brightness threshold segments them well enough to compare silhouettes.
    """
    out = {}
    for class_name in config.CLASS_NAMES:
        files = _class_files(class_name)
        chosen = rng.sample(files, min(SAMPLE_PER_CLASS, len(files)))
        ratios = []
        for path in chosen:
            gray = np.asarray(
                Image.open(path).convert("L").resize((128, 128)),
                dtype="float32")
            ratios.append(float((gray < 200).mean()))
        out[class_name] = {
            "mean": round(float(np.mean(ratios)), 4),
            "std": round(float(np.std(ratios)), 4),
            "p25": round(float(np.percentile(ratios, 25)), 4),
            "p75": round(float(np.percentile(ratios, 75)), 4),
        }
    return out


INTERPRETATIONS = {
    "class_balance":
        "The four classes are exactly balanced at 1,600 training images each. "
        "That means plain accuracy is an honest headline metric and no "
        "resampling or class weighting is needed — a skewed dataset would "
        "have forced both.",
    "channel_means":
        "Measured across the bean pixels only, longberry is clearly the "
        "lightest class (brightness 137 against 124-129 for the rest) and "
        "leads in all three channels. The revealing part is the bottom of the "
        "chart: defect (124.9) and peaberry (124.5) are almost indistinguishable "
        "in colour. Whatever separates those two classes, it is not colour — "
        "so the model has to find it in shape.",
    "area_ratios":
        "Bean area — the share of the frame the bean occupies — supplies "
        "exactly what colour could not. Peaberry is the most compact class "
        "(0.205) and also the most consistent (std 0.047), while premium is "
        "the largest (0.268). So peaberry, which colour could not separate "
        "from defect, is cleanly separable by size. Defect meanwhile has the "
        "second-widest spread (std 0.081) and sits mid-range on both measures.",
    "story":
        "Read together, the two feature charts predicted the model's weakness "
        "before training ran. Every other class owns a distinctive corner of "
        "the feature space: longberry is the lightest, peaberry the smallest "
        "and most consistent, premium the largest. Defect owns nothing — it "
        "sits mid-range in colour and mid-range in area, with wide variance in "
        "both, so it overlaps all three neighbours. That is precisely what the "
        "confusion matrix went on to show: defect had the lowest recall of any "
        "class at 0.767, against longberry's 0.930. The data said where the "
        "model would struggle, and it struggled there.",
}


def main():
    rng = random.Random(config.SEED)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "class_counts": class_counts(),
        "channel_means": channel_means(rng),
        "area_ratios": area_ratios(rng),
        "interpretations": INTERPRETATIONS,
    }
    config.INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.INSIGHTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {config.INSIGHTS_PATH}")


if __name__ == "__main__":
    main()

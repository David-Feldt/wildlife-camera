"""Fine-tune yolo11n on-device and stage the result for the detector.

Run on the Jetson with the tracker stopped (training wants the GPU and the
~6 GB of RAM the capture pipeline otherwise holds):

    python -m crittercam.training.finetune --epochs 100

Defaults assemble a dataset from the ENA24 starter pool, fine-tune from the
existing yolo11n.pt, and copy the winning weights to <data_root>/models/. The
Orin Nano (8 GB shared) trains at a small batch and is slow — hours, not the
minutes a desktop GPU takes — so this is an occasional batch job, not a loop.
After it finishes, point the detector at the new weights (see the printed next
steps): set detector.model and update detector.classes to the new class list.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from crittercam.config import load_config, setup_logging
from crittercam.training.dataset import build_yolo_dataset

log = logging.getLogger(__name__)

# ~/wildlife-camera-mock/source/ena24 — the labeled starter pool (see CLAUDE.md).
MOCK_ENA24 = Path.home() / "wildlife-camera-mock" / "source" / "ena24"


def finetune(
    data_yaml: Path,
    base_model: Path,
    project: Path,
    name: str,
    *,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    device: int = 0,
    workers: int = 2,
    patience: int = 20,
) -> Path:
    """Fine-tune `base_model` on `data_yaml`; return the path to best.pt.

    batch/imgsz/workers are deliberately conservative for the Orin's 8 GB; cache
    is off because the dataset will not fit in RAM alongside training."""
    from ultralytics import YOLO  # heavy import, keep local

    model = YOLO(str(base_model), task="detect")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        cache=False,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training finished but {best} is missing")
    return best


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.log_level)

    ap = argparse.ArgumentParser(description="Fine-tune yolo11n on-device (M5).")
    ap.add_argument("--coco", type=Path, default=MOCK_ENA24 / "subset.json",
                    help="COCO json with categories/images/annotations")
    ap.add_argument("--images", type=Path, default=MOCK_ENA24 / "images",
                    help="directory of source JPEGs")
    ap.add_argument("--base-model", type=Path, default=cfg.model_path,
                    help="weights to fine-tune from (default: configured yolo11n.pt)")
    ap.add_argument("--name", default="wildlife", help="output model/run name")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--skip-build", action="store_true",
                    help="reuse the dataset.yaml from a previous build")
    args = ap.parse_args()

    train_root = cfg.data_root / "training"
    data_yaml = train_root / "dataset" / "dataset.yaml"
    if args.skip_build:
        if not data_yaml.exists():
            ap.error(f"--skip-build but {data_yaml} does not exist; run once without it")
        log.info("reusing dataset %s", data_yaml)
    else:
        manifest = build_yolo_dataset(
            args.coco, args.images, train_root / "dataset",
            val_fraction=args.val_fraction,
        )
        data_yaml = manifest.yaml_path
        log.info("classes (%d): %s", len(manifest.names), ", ".join(manifest.names))

    best = finetune(
        data_yaml, args.base_model, train_root / "runs", args.name,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
    )

    staged = cfg.data_root / "models" / f"{args.name}.pt"
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, staged)
    log.info("staged best weights -> %s", staged)

    # Read back the trained class list so the config update is copy-pasteable.
    from ultralytics import YOLO
    names = list(YOLO(str(staged), task="detect").model.names.values())
    print("\n=== next steps ===")
    print(f"trained weights: {staged}")
    print("in <data_root>/config.yaml set:")
    print(f"  detector:\n    model: {args.name}.pt")
    print(f"    classes: {names}")
    print("(detector.classes must match the model's classes or startup validation fails)")


if __name__ == "__main__":
    main()

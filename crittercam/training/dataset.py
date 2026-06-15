"""Assemble a YOLO-format dataset from a COCO-style source.

The ENA24-detection subset (`~/wildlife-camera-mock/source/ena24/subset.json`)
is COCO: `categories`, `images` (id/file_name/width/height), `annotations`
(category_id + `bbox` as [x, y, w, h] in absolute pixels). ultralytics wants,
per split, an `images/` tree of JPEGs and a parallel `labels/` tree of `.txt`
files (one row `cls cx cy w h`, all normalized 0-1), plus a `dataset.yaml`.

We symlink images rather than copy (420+ JPEGs, same filesystem) and remap the
source category ids to a contiguous 0..N-1 space — YOLO requires dense class
ids, and the source skips some (e.g. Vehicle is excluded from the subset).

The split here is by image, which is correct for this off-camera starter pool.
Validation on *this camera's* sightings must instead split by day/clip (never by
frame — adjacent frames leak); that lives with the sighting-export path, not
here.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from random import Random

import yaml

log = logging.getLogger(__name__)


@dataclass
class DatasetManifest:
    yaml_path: Path
    names: list[str]
    train_count: int
    val_count: int
    skipped: int  # images referenced by the json whose JPEG was missing


def _coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # clamp: a few source boxes graze the frame edge and round just past it
    clamp = lambda v: min(1.0, max(0.0, v))
    return clamp(cx), clamp(cy), clamp(nw), clamp(nh)


def _reset_dir(d: Path) -> None:
    """Make `d` exist and empty (idempotent rebuilds; we only ever create
    symlinks/txt under here, so unlinking children is safe)."""
    d.mkdir(parents=True, exist_ok=True)
    for child in d.iterdir():
        if child.is_dir():
            for f in child.iterdir():
                f.unlink()
            child.rmdir()
        else:
            child.unlink()


def build_yolo_dataset(
    coco_json: Path,
    images_dir: Path,
    out_dir: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> DatasetManifest:
    """Convert a COCO json + image dir into a YOLO dataset under `out_dir`.

    Returns a manifest with the dataset.yaml path, class names, and counts.
    Images whose JPEG is missing on disk are skipped (counted, not fatal)."""
    coco = json.loads(Path(coco_json).read_text())

    # Contiguous class ids: keep only categories that actually carry annotations,
    # ordered by source id for a stable mapping across rebuilds.
    used_cat_ids = {a["category_id"] for a in coco["annotations"]}
    cats = sorted((c for c in coco["categories"] if c["id"] in used_cat_ids),
                  key=lambda c: c["id"])
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    anns_by_image: dict[str, list[dict]] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    images = list(coco["images"])
    Random(seed).shuffle(images)
    n_val = int(len(images) * val_fraction)
    splits = {"val": images[:n_val], "train": images[n_val:]}

    for sub in ("images", "labels"):
        for split in splits:
            _reset_dir(out_dir / sub / split)

    counts = {"train": 0, "val": 0}
    skipped = 0
    for split, imgs in splits.items():
        for img in imgs:
            src = Path(images_dir) / img["file_name"]
            if not src.exists():
                skipped += 1
                continue
            stem = Path(img["file_name"]).stem
            link = out_dir / "images" / split / img["file_name"]
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(src.resolve(), link)

            rows = []
            for a in anns_by_image.get(img["id"], []):
                cx, cy, w, h = _coco_bbox_to_yolo(a["bbox"], img["width"], img["height"])
                rows.append(f"{cat_to_idx[a['category_id']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            # an empty .txt is a valid YOLO "background" image, not an error
            (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(rows))
            counts[split] += 1

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }, sort_keys=False))

    log.info("dataset built: %d train, %d val, %d classes, %d skipped -> %s",
             counts["train"], counts["val"], len(names), skipped, yaml_path)
    return DatasetManifest(yaml_path, names, counts["train"], counts["val"], skipped)

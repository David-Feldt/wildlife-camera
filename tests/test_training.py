"""COCO -> YOLO dataset assembly for M5 fine-tuning.

Pure file/logic test, no GPU or ultralytics: the actual fine-tune is exercised
on the Orin (hours-long), which a unit test can't stand in for.
"""
import json

import yaml

from crittercam.training.dataset import build_yolo_dataset


def _write_coco(tmp_path, categories, images, annotations, make_files=True):
    coco = {"categories": categories, "images": images, "annotations": annotations}
    cj = tmp_path / "coco.json"
    cj.write_text(json.dumps(coco))
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    if make_files:
        for im in images:
            (img_dir / im["file_name"]).write_bytes(b"\xff\xd8\xff")  # stub jpeg
    return cj, img_dir


def test_build_basic_conversion_and_remap(tmp_path):
    # category ids are non-contiguous (3, 7) with an unused one (5) -> must remap
    # to a dense 0..N-1 ordered by source id, dropping the unused category.
    cats = [{"name": "CatB", "id": 7}, {"name": "CatA", "id": 3}, {"name": "Unused", "id": 5}]
    images = [
        {"id": "1", "file_name": "1.jpg", "width": 100, "height": 100},
        {"id": "2", "file_name": "2.jpg", "width": 200, "height": 100},
        {"id": "3", "file_name": "3.jpg", "width": 100, "height": 100},  # background
    ]
    anns = [
        {"id": "a", "image_id": "1", "category_id": 7, "bbox": [10, 20, 30, 40]},
        {"id": "b", "image_id": "2", "category_id": 3, "bbox": [0, 0, 100, 50]},
    ]
    cj, img_dir = _write_coco(tmp_path, cats, images, anns)

    m = build_yolo_dataset(cj, img_dir, tmp_path / "out", val_fraction=0.0, seed=1)

    assert m.names == ["CatA", "CatB"]      # id order 3, 7 -> idx 0, 1 (Unused dropped)
    assert (m.train_count, m.val_count, m.skipped) == (3, 0, 0)

    cfg = yaml.safe_load(m.yaml_path.read_text())
    assert cfg["names"] == {0: "CatA", 1: "CatB"}
    assert cfg["train"] == "images/train" and cfg["val"] == "images/val"

    labels = tmp_path / "out" / "labels" / "train"
    # cat 7 -> idx 1; 100x100, bbox[10,20,30,40] -> cx .25 cy .40 w .30 h .40
    assert labels.joinpath("1.txt").read_text() == "1 0.250000 0.400000 0.300000 0.400000"
    # cat 3 -> idx 0; 200x100, bbox[0,0,100,50] -> cx .25 cy .25 w .50 h .50
    assert labels.joinpath("2.txt").read_text() == "0 0.250000 0.250000 0.500000 0.500000"
    # background image -> empty label file (valid YOLO negative, not an error)
    assert labels.joinpath("3.txt").read_text() == ""
    # images are symlinked, not copied
    assert (tmp_path / "out" / "images" / "train" / "1.jpg").is_symlink()


def test_missing_image_is_skipped_not_fatal(tmp_path):
    cats = [{"name": "A", "id": 0}]
    images = [
        {"id": "1", "file_name": "present.jpg", "width": 10, "height": 10},
        {"id": "2", "file_name": "gone.jpg", "width": 10, "height": 10},
    ]
    anns = [{"id": "a", "image_id": "1", "category_id": 0, "bbox": [1, 1, 2, 2]}]
    cj, img_dir = _write_coco(tmp_path, cats, images, anns)
    (img_dir / "gone.jpg").unlink()  # referenced by json but absent on disk

    m = build_yolo_dataset(cj, img_dir, tmp_path / "out", val_fraction=0.0)
    assert m.skipped == 1 and m.train_count == 1


def test_split_is_deterministic_by_fraction(tmp_path):
    cats = [{"name": "A", "id": 0}]
    images = [{"id": str(i), "file_name": f"{i}.jpg", "width": 10, "height": 10}
              for i in range(10)]
    anns = [{"id": f"a{i}", "image_id": str(i), "category_id": 0, "bbox": [1, 1, 2, 2]}
            for i in range(10)]
    cj, img_dir = _write_coco(tmp_path, cats, images, anns)

    m = build_yolo_dataset(cj, img_dir, tmp_path / "out", val_fraction=0.3, seed=42)
    assert (m.train_count, m.val_count) == (7, 3)


def test_rebuild_is_idempotent(tmp_path):
    cats = [{"name": "A", "id": 0}]
    images = [{"id": "1", "file_name": "1.jpg", "width": 10, "height": 10}]
    anns = [{"id": "a", "image_id": "1", "category_id": 0, "bbox": [1, 1, 2, 2]}]
    cj, img_dir = _write_coco(tmp_path, cats, images, anns)

    build_yolo_dataset(cj, img_dir, tmp_path / "out", val_fraction=0.0)
    m = build_yolo_dataset(cj, img_dir, tmp_path / "out", val_fraction=0.0)  # again
    train_imgs = list((tmp_path / "out" / "images" / "train").iterdir())
    assert len(train_imgs) == 1 and m.train_count == 1  # no duplicate/stale links

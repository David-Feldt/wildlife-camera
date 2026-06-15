"""Milestone 5: on-device fine-tuning of yolo11n on backyard wildlife.

Unlike the tracker/web runtime, this package is a batch tool run by hand on the
Jetson (the GPU stack — torch 2.8.0 + CUDA 12.6 — is validated; see CLAUDE.md).
It never imports tracker/web modules; it only reads the same config for paths.

Pipeline: `dataset.build_yolo_dataset` assembles a YOLO-format dataset from a
COCO source (the ENA24 starter pool today; this camera's labeled sightings
later), then `finetune.finetune` fine-tunes yolo11n and drops best.pt where the
detector can pick it up.
"""

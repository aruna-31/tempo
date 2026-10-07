# Derived Release

This directory contains the privacy-aware derived annotation package used for evaluation-level reproducibility of the classroom distraction tracking paper.

This package can be provided as a supplemental file for the **self-curated evaluation dataset** because it documents the evaluation annotations while avoiding redistribution of identifiable classroom videos.

## Files

- `derived_status_boxes.csv`: per-frame bounding boxes, IDs, and status labels
- `manifest.json`: schema and video list metadata

## Dataset scope

The original evaluation data consist of seven authentic classroom videos internally named `class_1` to `class_7`. In response to reviewer requests for an openly auditable evaluation set, the seven self-collected classroom videos are publicly released for **research and non-commercial use only**, governed by `DATASET_LICENSE.md` in the repository root. See `DATA_AVAILABILITY.md` for the release channel and use terms.

This derived package provides non-visual annotation records extracted from the internal annotation files. It is intended to support code-level and evaluation-level verification without exposing raw images, video frames, audio, or source paths, and remains useful for users who only need the metric-level reproduction without downloading the raw videos.

## Record schema

Each row in `derived_status_boxes.csv` represents one annotated person instance in one frame:

- `video`: internal video identifier, such as `class_1`
- `frame_id`: 1-indexed frame number
- `id`: person identity ID within the video
- `x1`, `y1`, `x2`, `y2`: bounding box coordinates in pixels
- `status`: binary classroom-attention process label, either `Focused` or `Distracted`

The same schema is listed in `manifest.json`.

## Privacy statement

This package does not contain raw classroom videos, image frames, face images, audio, classroom location information, or original local file paths. It only contains derived annotation fields required for reproducing evaluation metrics. Any use of these annotations should avoid re-identification attempts, surveillance use, profiling, or punitive decision-making.

## Intended use

- reproduce evaluation metrics without distributing original classroom videos
- compare tracking / status methods using the same derived labels
- audit the reported frame-level annotation structure and evaluation protocol

## Out-of-scope use

- reconstructing or attempting to infer identifiable classroom imagery
- identifying individual students or classroom locations
- using the annotations for surveillance, profiling, or disciplinary decisions

## Regeneration

```bash
python export_derived_dataset.py --annotations_dir annotations --out_dir derived_release
```

The exporter reads `annotations/*_status_annotations.json` and writes the derived CSV and manifest. The exported package intentionally omits source video paths and visual data.

## Citation

If you use this derived annotation package, cite the corresponding paper and the public repository or release URL.

See also:

- `../DATA_AVAILABILITY.md`
- `../DATACARD.md`

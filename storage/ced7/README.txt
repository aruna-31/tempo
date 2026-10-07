CDED-7 — Classroom Distraction Evaluation Dataset (v1.1.0-dataset)
==================================================================

This archive bundles the version-of-record artifacts supporting the
manuscript "Interpretable classroom distraction estimation via
tracking-to-analysis continuity" (PeerJ Computer Science).

It is the canonical Zenodo deposit corresponding to the GitHub tag
v1.1.0-dataset of
  https://github.com/fangsheng223/classroom-distraction-tracker

Contents
--------
videos/
  class_1.mp4 ... class_7.mp4
    Seven self-collected classroom videos used as the evaluation set in
    the manuscript. Recorded with fixed surveillance-grade IP cameras
    mounted at instructor-station height at the School of Artificial
    Intelligence, Luoyang Normal University, China. No audio channel
    is included. See Section 6.5 (Dataset Construction Protocol) of
    the repository README for the full recording protocol, inclusion /
    exclusion criteria, and anonymisation safeguards.
    Each video file is the canonical raw footage released with this
    version-of-record; please consult the per-video
    `annotations/class_{1..7}_status_annotations.json` files and the
    manifest at `derived_release/manifest.json` for the exact duration,
    frame rate, resolution, and captured-system metadata of each clip.

derived_release/
  derived_status_boxes.csv
  manifest.json
  README.md
    Privacy-aware derived annotation package used for evaluation-level
    reproducibility. Contains per-frame bounding boxes, stable person
    IDs, and a binary attention label (Focused / Distracted) for all
    seven videos, plus a machine-readable manifest listing per-clip
    duration, frame rate, resolution, and other captured-system metadata.
    Raw frames, audio, face images, and original local file paths are
    intentionally omitted.

submission_assets/
  README.md
  figures/   Manuscript figures (Fig1 ... Fig8, FigB1 ... FigB5) in both
             PNG and PDF formats, including the supplementary appendix
             figures and source clips.
  tables/    Supplementary LaTeX table sources (Table1.tex, Table2.tex).

Licence
-------
The seven videos, the per-video annotations, and the derived
annotation package are released for research and non-commercial use
ONLY under the terms of DATASET_LICENSE.md in the parent repository:
  https://github.com/fangsheng223/classroom-distraction-tracker

Permitted uses: academic research, method development, reproducibility.
Prohibited uses: commercial use, re-identification, face recognition,
biometric identification, emotion scoring, surveillance, profiling,
punitive decision-making about specific individuals or classes.

Where this license and any other applicable terms disagree, the
stricter restriction applies. Withdrawal requests can be sent to the
corresponding author; affected footage will be removed from the public
release within a reasonable period.

Citation
--------
Please cite BOTH the PeerJ Computer Science manuscript AND this
Zenodo record (the DOI at the top of the record page) when using this
dataset. The corresponding source code, derived annotations, and the
tagged GitHub Release are mirrored at:
  https://github.com/fangsheng223/classroom-distraction-tracker/releases/tag/v1.1.0-dataset
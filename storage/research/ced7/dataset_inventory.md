# CDED-7 Dataset Inventory & Inspection Report

**Canonical Source**: CDED-7 ([Zenodo Record 21207208](https://zenodo.org/records/21207208))
**Official Repository**: [https://github.com/fangsheng223/classroom-distraction-tracker](https://github.com/fangsheng223/classroom-distraction-tracker)
**License**: Research and non-commercial use only (DATASET_LICENSE.md)

---

## 1. 50–70 Student Observation Audit Finding

> **Audit Verification**: Dataset does not provide a verified 50–70 simultaneously visible single-camera classroom.

- **Maximum simultaneous visible persons in any CDED-7 video**: **30** (measured in `class_7`).
- **Range across the 7 sessions**: 18 to 30 simultaneous visible persons.
- **Conclusion**: CDED-7 captures medium-density classroom sessions (18–30 students simultaneously observable from the podium camera angle), NOT 50–70 simultaneous students.

---

## 2. Per-Video Inspection Matrix

| Video ID | Resolution | FPS | Total Frames | Duration (s) | Annotated Frames | Unique IDs | Min Sim. | Median Sim. | Max Sim. | Back-Row Density | Annot. Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `class_1` | 1906x1080 | 30.0 | 450 | 15.0 | 449 | 19 | 18 | 19.0 | **19** | 0.0% | 99.78% |
| `class_2` | 1906x1080 | 30.0 | 450 | 15.0 | 449 | 20 | 18 | 19.0 | **20** | 0.0% | 99.78% |
| `class_3` | 1906x1080 | 30.0 | 533 | 17.77 | 532 | 19 | 18 | 18.0 | **19** | 0.0% | 99.81% |
| `class_4` | 1906x1080 | 30.0 | 367 | 12.23 | 366 | 19 | 18 | 18.0 | **19** | 0.0% | 99.73% |
| `class_5` | 1906x1080 | 30.0 | 443 | 14.77 | 442 | 19 | 19 | 19.0 | **19** | 0.0% | 99.77% |
| `class_6` | 1920x1080 | 30.0 | 300 | 10.0 | 298 | 29 | 28 | 29.0 | **29** | 0.0% | 99.33% |
| `class_7` | 1920x1080 | 30.0 | 300 | 10.0 | 299 | 30 | 30 | 30.0 | **30** | 0.0% | 99.67% |

---

## 3. Label and Box Geometry Distribution

| Video ID | Focused Boxes | Distracted Boxes | Total Annotations | Median Box Width | Median Box Height | Median Area (px²) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `class_1` | 6447 | 2065 | 8512 | 171.0 px | 134.0 px | 24696.0 px² |
| `class_2` | 5665 | 2925 | 8590 | 176.0 px | 131.0 px | 24912.0 px² |
| `class_3` | 6593 | 3246 | 9839 | 199.0 px | 136.0 px | 29512.0 px² |
| `class_4` | 3970 | 2685 | 6655 | 173.0 px | 138.0 px | 26358.0 px² |
| `class_5` | 5120 | 3278 | 8398 | 179.0 px | 153.0 px | 28611.0 px² |
| `class_6` | 6763 | 1761 | 8524 | 111.0 px | 113.0 px | 12376.0 px² |
| `class_7` | 6940 | 2030 | 8970 | 119.0 px | 107.5 px | 12178.0 px² |

---

## 4. Methodological Separation Notes

- **Label Semantics**: CDED-7 provides binary `Focused` vs `Distracted` annotations on head-shoulder bounding boxes. In accordance with Section 3 of the research protocol, these are **NOT** mapped to TEMPO's 5 behavior classes (`Looking_Toward_Instruction`, `Reading`, `Writing`, `Peer_Interaction`, `Looking_Away`).
- **Evaluation Scope**: CDED-7 is strictly utilized as an external benchmark for single-camera person detection, back-row/small-person localization, spatial zone breakdown, and anonymous multi-target tracking continuity.

"""
Builds the real-data training dataset for the TEMPO temporal behaviour model.

Sources (REAL, publicly published, human-annotated classroom behaviour datasets):
  1. SCB5-Handrise-Read-write-2024-9-17  (Whiffe/SCB-dataset, arXiv:2304.02488)
        classes: 0=hand-raising, 1=read, 2=write
  2. SCB5-Discuss-2024-9-17              (same source)
        classes: 0=discuss
  3. SCB_BowTurnHead_20250509            (same source)
        classes: 0=BowHead, 1=TurnHead
  4. SCBehavior (CCNUZFW/SCBehavior, YOLO format)
        classes: 0=write, 1=read, 2=lookup, 3=turn_head, 4=raise_hand,
                 5=stand, 6=discuss

Every crop inherits the REAL human-provided label of its bounding box.
No synthetic data, no augmentation-generated labels, no class invention.

Seat grouping: within one dataset+split the images are numbered frames of
source videos. Frames are split into RUNS of consecutive numbers (one source
clip each); inside a run, a "seat" is a grid bin of normalized box centers
(seated students keep their center in one bin). Crops of one (run, seat) form
the temporal group that training samples 16-crop windows from — mirroring the
production path (ByteTrack track -> per-student crops -> 16-frame windows).

Training uses PER-TIMESTEP supervision: every crop of a window is supervised
with its own human label, so brief actions (hand-raise, head-turn) are learned
from the frames where humans marked them — no majority-purity requirement and
no invented labels.

Mapping to TEMPO's five OBSERVABLE classes (models/model_metadata.json):
  Looking_Toward_Instruction <- lookup, hand-raising (facing the board/teacher)
  Reading                    <- read
  Writing                    <- write
  Peer_Interaction           <- discuss
  Looking_Away               <- turn_head, BowHead, TurnHead, stand

Output (models/training_data/dataset):
  class_<idx>_<name>/  224x224 crop jpgs
  index.csv            relative_path,label,source,split,seat_group,frame_num
"""
import csv
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TD = os.path.join(BASE, "models", "training_data")
OUT = os.path.join(TD, "dataset")
IMG_SIZE = 224

TEMPO_CLASSES = [
    "Looking_Toward_Instruction",
    "Reading",
    "Writing",
    "Peer_Interaction",
    "Looking_Away",
]
TEMPO_IDX = {c: i for i, c in enumerate(TEMPO_CLASSES)}

SCB_MAP = {
    "hand-raising": "Looking_Toward_Instruction",
    "read": "Reading",
    "write": "Writing",
    "discuss": "Peer_Interaction",
    "BowHead": "Looking_Away",
    "TurnHead": "Looking_Away",
}
SCBEHAVIOR_MAP = {
    "lookup": "Looking_Toward_Instruction",
    "raise_hand": "Looking_Toward_Instruction",
    "read": "Reading",
    "write": "Writing",
    "discuss": "Peer_Interaction",
    "turn_head": "Looking_Away",
    "stand": "Looking_Away",
}

SCB_SOURCES = [
    (os.path.join(TD, "handrise", "SCB5-Handrise-Read-write-2024-9-17"),
     ["hand-raising", "read", "write"], "hrw"),
    (os.path.join(TD, "discuss", "SCB5-Discuss-2024-9-17"),
     ["discuss"], "disc"),
    (os.path.join(TD, "bowturnhead", "SCB_BowTurnHead_20250509", "SCB5-Turn-Bow-Head-2024-9-17"),
     ["BowHead", "TurnHead"], "bow"),
]


def crop_letterbox(img: np.ndarray, norm_box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    bx, by, bw, bh = norm_box
    h, w = img.shape[:2]
    x1 = int(max(0.0, (bx - bw / 2) * w))
    y1 = int(max(0.0, (by - bh / 2) * h))
    x2 = int(min(float(w), (bx + bw / 2) * w))
    y2 = int(min(float(h), (by + bh / 2) * h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    scale = min(IMG_SIZE / cw, IMG_SIZE / ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    dx, dy = (IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas


def frame_number(fn: str) -> int:
    return int("".join(c for c in os.path.splitext(fn)[0] if c.isdigit()) or 0)


def runs_of(nums: List[int]) -> List[List[int]]:
    """Split sorted frame numbers into runs of consecutive integers."""
    runs: List[List[int]] = []
    cur: List[int] = []
    for n in nums:
        if cur and n - cur[-1] != 1:
            runs.append(cur)
            cur = []
        cur.append(n)
    if cur:
        runs.append(cur)
    return runs


def process_split(img_dir: str, lbl_dir: str, tempo_of, prefix: str, split: str,
                  rows, bin_size: Tuple[float, float]) -> None:
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    fn_by_num = {frame_number(f): f for f in files}
    # Load per-frame boxes once
    boxes_by_num: Dict[int, List[Tuple[Tuple[float, float, float, float], str]]] = {}
    for n, fn in fn_by_num.items():
        lbl_path = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
        if not os.path.exists(lbl_path):
            continue
        with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(100)
        if "git-lfs" in head:
            continue
        lst = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                tempo = tempo_of(int(float(p[0])))
                if tempo is None:
                    continue
                lst.append((tuple(float(v) for v in p[1:5]), tempo))
        if lst:
            boxes_by_num[n] = lst

    BX, BY = bin_size
    n_crops = 0
    img_cache: Dict[str, np.ndarray] = {}
    for run in runs_of(sorted(boxes_by_num)):
        # seat bins within this run
        seat_obs: Dict[Tuple[int, int], List[Tuple[int, Tuple, str]]] = defaultdict(list)
        for n in run:
            for box, tempo in boxes_by_num[n]:
                seat_obs[(round(box[0] / BX), round(box[1] / BY))].append((n, box, tempo))
        for (sx, sy), obs in sorted(seat_obs.items()):
            gid = f"{prefix}|{split}|r{run[0]}|s{sx}_{sy}"
            for n, box, tempo in obs:
                fn = fn_by_num[n]
                if fn not in img_cache:
                    img_cache = {fn: cv2.imread(os.path.join(img_dir, fn))}  # keep cache small
                img = img_cache.get(fn)
                if img is None:
                    continue
                crop = crop_letterbox(img, box)
                if crop is None:
                    continue
                out_dir = os.path.join(OUT, f"class_{TEMPO_IDX[tempo]}_{tempo}")
                os.makedirs(out_dir, exist_ok=True)
                out_name = f"{prefix}_{split}_f{n:08d}_{sx}_{sy}.jpg"
                cv2.imwrite(os.path.join(out_dir, out_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append((f"class_{TEMPO_IDX[tempo]}_{tempo}/{out_name}", tempo, prefix, split, gid, str(n)))
                n_crops += 1
    print(f"  {prefix} {split}: crops={n_crops}")


def main() -> None:
    if os.path.isdir(OUT):
        for d in os.listdir(OUT):
            p = os.path.join(OUT, d)
            if os.path.isdir(p):
                for fn in os.listdir(p):
                    os.remove(os.path.join(p, fn))
    os.makedirs(OUT, exist_ok=True)
    rows: List[Tuple[str, str, str, str, str, str]] = []

    print("== SCB (Whiffe) datasets:")
    for root, class_names, prefix in SCB_SOURCES:
        def tempo_of(cls_id, names=class_names):
            if 0 <= cls_id < len(names):
                return SCB_MAP[names[cls_id]]
            return None
        bin_size = (0.08, 0.16) if prefix == "bow" else (0.06, 0.12)
        for split in ("train", "val"):
            process_split(os.path.join(root, "images", split),
                          os.path.join(root, "labels", split),
                          tempo_of, prefix, split, rows, bin_size)

    print("== SCBehavior (CCNUZFW) dataset:")
    yolo_root = os.path.join(TD, "scbehavior_dl")
    if os.path.isdir(yolo_root):
        names = list(SCBEHAVIOR_MAP.keys())

        def tempo_of_scb(cls_id):
            if 0 <= cls_id < len(names):
                return SCBEHAVIOR_MAP[names[cls_id]]
            return None
        for split in ("train", "val"):
            process_split(os.path.join(yolo_root, "images", split),
                          os.path.join(yolo_root, "labels", split),
                          tempo_of_scb, "scb", split, rows, (0.06, 0.12))
    else:
        print("  !! run scripts/fetch_scbehavior.py first")

    idx_path = os.path.join(OUT, "index.csv")
    with open(idx_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "label", "source", "split", "seat_group", "frame_num"])
        w.writerows(rows)

    per_split = Counter((r[1], r[3]) for r in rows)
    groups = Counter(r[4] for r in rows)
    print(f"\nTotal crops: {len(rows)} -> {idx_path}")
    print(f"seat groups: {len(groups)} (>=16 crops: {sum(1 for v in groups.values() if v >= 16)})")
    print("Per TEMPO class:")
    for c in TEMPO_CLASSES:
        tr = per_split.get((c, "train"), 0)
        va = per_split.get((c, "val"), 0)
        print(f"  {c:32s} train={tr:6d} val={va:6d}")


if __name__ == "__main__":
    sys.exit(main())

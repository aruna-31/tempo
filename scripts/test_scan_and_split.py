import os
import csv
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

BASE_DIR = os.path.abspath(".")
DATA_DIR = os.path.join(BASE_DIR, "models", "training_data")

TEMPO_CLASSES = [
    "Looking_Toward_Instruction",
    "Reading",
    "Writing",
    "Peer_Interaction",
    "Looking_Away",
]
TEMPO_TO_IDX = {c: i for i, c in enumerate(TEMPO_CLASSES)}

SCB_MAP = {
    "hand-raising": "Looking_Toward_Instruction",
    "read": "Reading",
    "write": "Writing",
    "discuss": "Peer_Interaction",
    "TurnHead": "Looking_Away",
    # BowHead excluded
}

SCBEHAVIOR_MAP = {
    "lookup": "Looking_Toward_Instruction",
    "raise_hand": "Looking_Toward_Instruction",
    "read": "Reading",
    "write": "Writing",
    "discuss": "Peer_Interaction",
    "turn_head": "Looking_Away",
    # stand excluded
}

def parse_video_and_frame(base: str, source: str) -> Tuple[str, int]:
    if source == "scb":
        num = int("".join(c for c in base if c.isdigit()) or 0)
        return "scbehavior_session_01", num
    if "_" in base:
        parts = base.split("_")
        return "scb_vid_%04d" % int(parts[0]), int(parts[1])
    if len(base) in (7, 8) and base.isdigit():
        return "scb_vid_%04d" % int(base[:4]), int(base[4:])
    if len(base) == 6 and base.isdigit():
        return "scb_vid_%04d" % int(base[:4]), int(base[4:])
    num = int("".join(c for c in base if c.isdigit()) or 0)
    return "scb_vid_%04d" % num, num

def box_iou(b1, b2):
    x1_min, x1_max = b1[0] - b1[2]/2, b1[0] + b1[2]/2
    y1_min, y1_max = b1[1] - b1[3]/2, b1[1] + b1[3]/2
    x2_min, x2_max = b2[0] - b2[2]/2, b2[0] + b2[2]/2
    y2_min, y2_max = b2[1] - b2[3]/2, b2[1] + b2[3]/2
    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_w * inter_h
    union_area = b1[2]*b1[3] + b2[2]*b2[3] - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def scan():
    scb_sources = [
        (os.path.join(DATA_DIR, "handrise", "SCB5-Handrise-Read-write-2024-9-17"),
         ["hand-raising", "read", "write"], "hrw"),
        (os.path.join(DATA_DIR, "discuss", "SCB5-Discuss-2024-9-17"),
         ["discuss"], "disc"),
        (os.path.join(DATA_DIR, "bowturnhead", "SCB_BowTurnHead_20250509", "SCB5-Turn-Bow-Head-2024-9-17"),
         ["BowHead", "TurnHead"], "bow"),
    ]

    seen_frames = {} # base -> list of (tempo_label, orig_label, bbox, img_path)
    duplicate_raw_frames = 0
    excluded_bowhead = 0
    excluded_stand = 0
    conflicting_boxes_dropped = 0

    for root, names, prefix in scb_sources:
        for split in ["train", "val"]:
            img_dir = os.path.join(root, "images", split)
            lbl_dir = os.path.join(root, "labels", split)
            if not os.path.exists(lbl_dir): continue
            for lbl_fn in os.listdir(lbl_dir):
                if not lbl_fn.endswith(".txt"): continue
                base = os.path.splitext(lbl_fn)[0]
                img_path = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    cand = os.path.join(img_dir, base + ext)
                    if os.path.exists(cand):
                        img_path = cand
                        break
                if not img_path: continue

                is_dup_frame = (base in seen_frames)
                if is_dup_frame:
                    duplicate_raw_frames += 1
                else:
                    seen_frames[base] = []

                lbl_path = os.path.join(lbl_dir, lbl_fn)
                with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(float(parts[0]))
                            if 0 <= cid < len(names):
                                orig_label = names[cid]
                                if orig_label == "BowHead":
                                    excluded_bowhead += 1
                                    continue
                                if orig_label not in SCB_MAP:
                                    continue
                                tempo_label = SCB_MAP[orig_label]
                                bbox = [float(p) for p in parts[1:5]]

                                # Check IoU with existing boxes on this frame
                                has_conflict = False
                                for ex_tempo, ex_orig, ex_box, _ in seen_frames[base]:
                                    if box_iou(bbox, ex_box) > 0.4:
                                        has_conflict = True
                                        conflicting_boxes_dropped += 1
                                        break
                                if not has_conflict:
                                    seen_frames[base].append((tempo_label, orig_label, bbox, img_path))

    # Scan CCNUZFW
    scb_root = os.path.join(DATA_DIR, "scbehavior_dl")
    scb_records = []
    if os.path.exists(scb_root):
        scb_names = list(SCBEHAVIOR_MAP.keys()) + ["stand"]
        for split in ["train", "val"]:
            img_dir = os.path.join(scb_root, "images", split)
            lbl_dir = os.path.join(scb_root, "labels", split)
            if not os.path.exists(lbl_dir): continue
            for lbl_fn in os.listdir(lbl_dir):
                if not lbl_fn.endswith(".txt"): continue
                base = os.path.splitext(lbl_fn)[0]
                img_path = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    cand = os.path.join(img_dir, base + ext)
                    if os.path.exists(cand):
                        img_path = cand
                        break
                if not img_path: continue
                lbl_path = os.path.join(lbl_dir, lbl_fn)
                with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(float(parts[0]))
                            if 0 <= cid < len(scb_names):
                                orig_label = scb_names[cid]
                                if orig_label == "stand":
                                    excluded_stand += 1
                                    continue
                                if orig_label not in SCBEHAVIOR_MAP:
                                    continue
                                tempo_label = SCBEHAVIOR_MAP[orig_label]
                                bbox = [float(p) for p in parts[1:5]]
                                vid, fnum = parse_video_and_frame(base, "scb")
                                scb_records.append({
                                    "video_id": vid,
                                    "frame_num": fnum,
                                    "img_path": img_path,
                                    "bbox": bbox,
                                    "original_label": orig_label,
                                    "tempo_label": tempo_label,
                                    "source": "scb",
                                })

    # Assemble SCB records from seen_frames
    scb5_records = []
    for base, items in seen_frames.items():
        vid, fnum = parse_video_and_frame(base, "hrw")
        for tempo_label, orig_label, bbox, img_path in items:
            scb5_records.append({
                "video_id": vid,
                "frame_num": fnum,
                "img_path": img_path,
                "bbox": bbox,
                "original_label": orig_label,
                "tempo_label": tempo_label,
                "source": "scb5",
            })

    all_records = scb5_records + scb_records
    print(f"Total unique raw frames in SCB: {len(seen_frames):,}")
    print(f"Duplicate frame instances removed: {duplicate_raw_frames:,}")
    print(f"Excluded BowHead samples: {excluded_bowhead:,}")
    print(f"Excluded stand samples: {excluded_stand:,}")
    print(f"Conflicting duplicate bounding boxes dropped: {conflicting_boxes_dropped:,}")
    print(f"Total valid bounding-box annotations: {len(all_records):,}")

    # Unique videos
    vids = sorted(set(r["video_id"] for r in all_records))
    print(f"Total Unique Source Videos: {len(vids)}")
    cls_cnt = Counter(r["tempo_label"] for r in all_records)
    print("Crops per class in all records:", dict(cls_cnt))
    return all_records, duplicate_raw_frames, excluded_bowhead

if __name__ == "__main__":
    scan()

#!/usr/bin/env python3
"""
TEMPO — Dataset Verification Script
Verifies video split integrity, crop counts, and 16-frame sequence counts.
"""
import csv
import json
import os
from collections import Counter, defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(BASE_DIR, "models", "training_data", "dataset", "index.csv")
SEQ_PATH = os.path.join(BASE_DIR, "models", "training_data", "dataset", "sequences.json")

def main():
    rows = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    with open(SEQ_PATH, "r", encoding="utf-8") as f:
        seq_data = json.load(f)

    videos_by_split = defaultdict(set)
    crops_by_class_split = Counter()
    crops_by_class = Counter()

    for r in rows:
        vid = r["source_video_id"]
        sp = r["split"]
        cls = r["tempo_label"]
        videos_by_split[sp].add(vid)
        crops_by_class_split[(cls, sp)] += 1
        crops_by_class[cls] += 1

    all_vids = set().union(*videos_by_split.values())

    print("=" * 80)
    print("  ACTUAL SMART CLASSROOM DATASET VERIFICATION STATISTICS")
    print("=" * 80)
    print(f"1. Actual Smart Classroom Videos Processed: {len(all_vids)}")
    print(f"   Sample Video IDs: {sorted(all_vids)[:8]} ... (+{len(all_vids)-8} more)")

    print(f"\n2. Train / Val / Test Video Counts:")
    n_tot = len(all_vids)
    tr_v = videos_by_split["train"]
    va_v = videos_by_split["val"]
    te_v = videos_by_split["test"]
    print(f"   - Train : {len(tr_v):>2d} videos ({len(tr_v)/n_tot*100:.1f}%)")
    print(f"   - Val   : {len(va_v):>2d} videos ({len(va_v)/n_tot*100:.1f}%)")
    print(f"   - Test  : {len(te_v):>2d} videos ({len(te_v)/n_tot*100:.1f}%)")

    classes = [
        "Looking_Toward_Instruction",
        "Reading",
        "Writing",
        "Peer_Interaction",
        "Looking_Away",
    ]
    print(f"\n3. Crops per TEMPO Class (Total: {len(rows):,}):")
    print(f"   {'TEMPO Class':<32} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("   " + "-" * 74)
    for c in classes:
        tr = crops_by_class_split[(c, "train")]
        va = crops_by_class_split[(c, "val")]
        te = crops_by_class_split[(c, "test")]
        tot = crops_by_class[c]
        print(f"   {c:<32} {tr:>10,d} {va:>10,d} {te:>10,d} {tot:>10,d}")

    print(f"\n4. Valid 16-Frame Sequences per Class (Total: {sum(len(v) for v in seq_data.values()):,}):")
    seq_splits = defaultdict(Counter)
    for c, seq_list in seq_data.items():
        for s in seq_list:
            seq_splits[s["split"]][c] += 1

    print(f"   {'TEMPO Class':<32} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("   " + "-" * 74)
    for c in classes:
        tr = seq_splits["train"][c]
        va = seq_splits["val"][c]
        te = seq_splits["test"][c]
        tot = len(seq_data.get(c, []))
        print(f"   {c:<32} {tr:>10,d} {va:>10,d} {te:>10,d} {tot:>10,d}")

    print(f"\n5. Confirmation that No Video Appears in Multiple Splits:")
    inter_tr_va = tr_v & va_v
    inter_tr_te = tr_v & te_v
    inter_va_te = va_v & te_v
    print(f"   - Train & Val  video intersection: {len(inter_tr_va)}")
    print(f"   - Train & Test video intersection: {len(inter_tr_te)}")
    print(f"   - Val   & Test video intersection: {len(inter_va_te)}")
    assert len(inter_tr_va) == 0 and len(inter_tr_te) == 0 and len(inter_va_te) == 0
    print("   [CONFIRMED] Zero video overlap across Train, Val, and Test.")
    print("=" * 80)

if __name__ == "__main__":
    main()

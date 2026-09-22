import os, json, csv, random
from collections import Counter, defaultdict

def test_split_and_seq():
    from test_scan_and_split import scan, parse_video_and_frame
    all_records, dupes, bow_ex = scan()

    # Assign seat tracks
    bx, by = 0.07, 0.14
    for r in all_records:
        cx, cy = r["bbox"][0], r["bbox"][1]
        grid_x = int(round(cx / bx))
        grid_y = int(round(cy / by))
        r["student_track_id"] = f"{r['video_id']}_s{grid_x}_{grid_y}"

    # Partition 70/15/15
    video_classes = defaultdict(Counter)
    for r in all_records:
        video_classes[r["video_id"]][r["tempo_label"]] += 1

    all_vids = sorted(video_classes.keys())
    n_total = len(all_vids)
    n_train = 119
    n_val = 25
    n_test = 25

    best_seed = 42
    best_score = -1
    for seed in range(500):
        rng = random.Random(seed)
        shuffled = list(all_vids)
        rng.shuffle(shuffled)
        tr = shuffled[:n_train]
        va = shuffled[n_train:n_train+n_val]
        te = shuffled[n_train+n_val:]

        tr_cnt = Counter()
        for v in tr: tr_cnt.update(video_classes[v])
        va_cnt = Counter()
        for v in va: va_cnt.update(video_classes[v])
        te_cnt = Counter()
        for v in te: te_cnt.update(video_classes[v])

        if len(tr_cnt) == 5 and len(va_cnt) == 5 and len(te_cnt) == 5:
            min_c = min(min(tr_cnt.values()), min(va_cnt.values()), min(te_cnt.values()))
            if min_c > best_score:
                best_score = min_c
                best_seed = seed

    rng = random.Random(best_seed)
    shuffled = list(all_vids)
    rng.shuffle(shuffled)
    train_vids = set(shuffled[:n_train])
    val_vids = set(shuffled[n_train:n_train+n_val])
    test_vids = set(shuffled[n_train+n_val:])

    assert not (train_vids & val_vids)
    assert not (train_vids & test_vids)
    assert not (val_vids & test_vids)

    for r in all_records:
        v = r["video_id"]
        if v in train_vids: r["split"] = "train"
        elif v in val_vids: r["split"] = "val"
        else: r["split"] = "test"

    print("Partition successful:")
    print("Train videos:", len(train_vids), "Val videos:", len(val_vids), "Test videos:", len(test_vids))

    # Test sequence generation with stride 16
    tracks = defaultdict(list)
    for r in all_records:
        tracks[r["student_track_id"]].append(r)

    seq_counts = defaultdict(Counter)
    total_seqs = 0
    SEQ_LEN = 16
    stride = 16 # Non-overlapping!

    for track_id, items in tracks.items():
        if len(items) < SEQ_LEN: continue
        # Deduplicate multiple items with exact same frame_num in the same tracklet
        seen_fnums = set()
        unique_items = []
        # Sort by frame_num
        items.sort(key=lambda x: int(x["frame_num"]))
        for it in items:
            fn = int(it["frame_num"])
            if fn not in seen_fnums:
                seen_fnums.add(fn)
                unique_items.append(it)

        if len(unique_items) < SEQ_LEN: continue

        split = unique_items[0]["split"]
        for s in range(0, len(unique_items) - SEQ_LEN + 1, stride):
            window = unique_items[s:s+SEQ_LEN]
            fnums = [int(w["frame_num"]) for w in window]
            is_cont = all(1 <= fnums[i+1] - fnums[i] <= 3 for i in range(len(fnums)-1))
            if not is_cont: continue
            labels = [w["tempo_label"] for w in window]
            if len(set(labels)) != 1: continue
            seq_counts[split][labels[0]] += 1
            total_seqs += 1

    print(f"Total non-overlapping (stride=16) sequences: {total_seqs}")
    for sp in ["train", "val", "test"]:
        print(f"Split {sp}:", dict(seq_counts[sp]))

if __name__ == "__main__":
    test_split_and_seq()

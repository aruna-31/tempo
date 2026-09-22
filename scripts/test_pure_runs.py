import sys; sys.path.append('scripts')
import test_scan_and_split
from collections import Counter, defaultdict

all_records, _, _ = test_scan_and_split.scan()

# Group by student_track_id
tracks = defaultdict(list)
for r in all_records:
    tid = f"{r['video_id']}_s{int(round(r['bbox'][0]/0.07))}_{int(round(r['bbox'][1]/0.14))}"
    tracks[tid].append(r)

SEQ_LEN = 16
stride = 16

pure_continuous_seqs = defaultdict(list)

for tid, items in tracks.items():
    # Deduplicate same frame_num
    seen_fnums = set()
    u_items = []
    items.sort(key=lambda x: int(x["frame_num"]))
    for it in items:
        fn = int(it["frame_num"])
        if fn not in seen_fnums:
            seen_fnums.add(fn)
            u_items.append(it)

    if len(u_items) < SEQ_LEN:
        continue

    # Partition tracklet into continuous, single-label runs
    runs = []
    current_run = [u_items[0]]
    for i in range(len(u_items) - 1):
        prev = u_items[i]
        curr = u_items[i+1]
        diff = int(curr["frame_num"]) - int(prev["frame_num"])
        same_label = (curr["tempo_label"] == prev["tempo_label"])
        if (1 <= diff <= 3) and same_label:
            current_run.append(curr)
        else:
            if len(current_run) >= SEQ_LEN:
                runs.append(current_run)
            current_run = [curr]
    if len(current_run) >= SEQ_LEN:
        runs.append(current_run)

    # For each run, extract non-overlapping 16-frame windows (stride=16)
    for run in runs:
        for s in range(0, len(run) - SEQ_LEN + 1, stride):
            window = run[s : s + SEQ_LEN]
            label = window[0]["tempo_label"]
            vid = window[0]["video_id"]
            pure_continuous_seqs[label].append({
                "track_id": tid,
                "video_id": vid,
                "class_name": label,
                "start_frame": int(window[0]["frame_num"]),
                "end_frame": int(window[-1]["frame_num"]),
            })

print("Total non-overlapping (stride=16) pure continuous sequences:")
for c, seqs in sorted(pure_continuous_seqs.items()):
    v_cnt = len(set(s["video_id"] for s in seqs))
    print(f"  {c:<30}: {len(seqs):>5d} sequences across {v_cnt:>2d} videos")
print("Total sequences:", sum(len(v) for v in pure_continuous_seqs.values()))

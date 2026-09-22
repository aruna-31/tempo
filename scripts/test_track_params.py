import sys; sys.path.append('scripts')
import test_scan_and_split
from collections import defaultdict, Counter

all_records, _, _ = test_scan_and_split.scan()

# Group by student_track_id
# Let's test different track assignment strategies:
# Strategy A: grid binning bx=0.07, by=0.14
# Strategy B: grid binning bx=0.04, by=0.08 (finer grid so adjacent students don't merge)
# Strategy C: nearest center tracking

for grid_name, bx, by in [('Coarse (0.07, 0.14)', 0.07, 0.14), ('Finer (0.04, 0.08)', 0.04, 0.08), ('Ultra-fine (0.03, 0.06)', 0.03, 0.06)]:
    tracks = defaultdict(list)
    for r in all_records:
        tid = f"{r['video_id']}_s{int(round(r['bbox'][0]/bx))}_{int(round(r['bbox'][1]/by))}"
        tracks[tid].append(r)

    for max_gap in [3, 5, 8]:
        seq_cnt = Counter()
        for tid, items in tracks.items():
            seen_f = set()
            u_items = []
            items.sort(key=lambda x: int(x["frame_num"]))
            for it in items:
                fn = int(it["frame_num"])
                if fn not in seen_f:
                    seen_f.add(fn)
                    u_items.append(it)
            if len(u_items) < 16: continue

            # Extract continuous pure runs
            runs = []
            cur_run = [u_items[0]]
            for i in range(len(u_items)-1):
                prev = u_items[i]
                curr = u_items[i+1]
                diff = int(curr["frame_num"]) - int(prev["frame_num"])
                same_l = (curr["tempo_label"] == prev["tempo_label"])
                if (1 <= diff <= max_gap) and same_l:
                    cur_run.append(curr)
                else:
                    if len(cur_run) >= 16: runs.append(cur_run)
                    cur_run = [curr]
            if len(cur_run) >= 16: runs.append(cur_run)

            for run in runs:
                for s in range(0, len(run)-16+1, 16):
                    seq_cnt[run[s]["tempo_label"]] += 1

        print(f"Grid: {grid_name}, max_gap={max_gap} -> Total non-overlapping (stride=16) seqs: {sum(seq_cnt.values())}")
        print("  Breakdown:", dict(seq_cnt))

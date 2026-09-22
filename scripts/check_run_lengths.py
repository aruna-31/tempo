import sys; sys.path.append('scripts')
import test_scan_and_split
from collections import defaultdict, Counter

all_records, _, _ = test_scan_and_split.scan()

tracks = defaultdict(list)
for r in all_records:
    tid = f"{r['video_id']}_s{int(round(r['bbox'][0]/0.07))}_{int(round(r['bbox'][1]/0.14))}"
    tracks[tid].append(r)

run_lens_by_class = defaultdict(list)

for tid, items in tracks.items():
    seen_f = set()
    u_items = []
    items.sort(key=lambda x: int(x["frame_num"]))
    for it in items:
        fn = int(it["frame_num"])
        if fn not in seen_f:
            seen_f.add(fn)
            u_items.append(it)

    cur_run = [u_items[0]]
    for i in range(len(u_items)-1):
        prev = u_items[i]
        curr = u_items[i+1]
        diff = int(curr["frame_num"]) - int(prev["frame_num"])
        if (1 <= diff <= 5) and (curr["tempo_label"] == prev["tempo_label"]):
            cur_run.append(curr)
        else:
            run_lens_by_class[cur_run[0]["tempo_label"]].append(len(cur_run))
            cur_run = [curr]
    run_lens_by_class[cur_run[0]["tempo_label"]].append(len(cur_run))

for c in sorted(run_lens_by_class.keys()):
    lens = run_lens_by_class[c]
    over16 = [l for l in lens if l >= 16]
    print(f"Class: {c:<28} | Total runs: {len(lens):>5d} | Max run: {max(lens):>3d} | Runs >= 16: {len(over16):>4d} | Median run: {sorted(lens)[len(lens)//2]}")

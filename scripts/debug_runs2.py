import sys; sys.path.append('scripts')
import test_scan_and_split

all_records, _, _ = test_scan_and_split.scan()

for test_vid in ['scb_vid_0086', 'scbehavior_session_01', 'scb_vid_3000']:
    rec = [r for r in all_records if r['video_id'] == test_vid]
    tracks = {}
    for r in rec:
        tid = f"{r['video_id']}_s{int(round(r['bbox'][0]/0.07))}_{int(round(r['bbox'][1]/0.14))}"
        tracks.setdefault(tid, []).append(r)

    print(f"\nVideo: {test_vid} ({len(rec)} records, {len(tracks)} tracks)")
    n_seqs = 0
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

        # Check why
        runs = []
        cur_run = [u_items[0]]
        for i in range(len(u_items)-1):
            prev = u_items[i]
            curr = u_items[i+1]
            diff = int(curr["frame_num"]) - int(prev["frame_num"])
            same_l = (curr["tempo_label"] == prev["tempo_label"])
            if (1 <= diff <= 3) and same_l:
                cur_run.append(curr)
            else:
                if len(cur_run) >= 16: runs.append(cur_run)
                cur_run = [curr]
        if len(cur_run) >= 16: runs.append(cur_run)

        for run in runs:
            n_seqs += len(run) // 16

    print(f"  Generated {n_seqs} non-overlapping (stride=16) sequences!")

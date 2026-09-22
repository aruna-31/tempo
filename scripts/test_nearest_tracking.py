import sys; sys.path.append('scripts')
import test_scan_and_split
from collections import defaultdict, Counter
import math

all_records, _, _ = test_scan_and_split.scan()

def track_video_by_distance(records):
    # Group by frame
    by_frame = defaultdict(list)
    for r in records:
        by_frame[int(r["frame_num"])].append(r)

    active_tracks = [] # (tid, last_fn, last_cx, last_cy, list_of_records)
    finished_tracks = []
    next_tid = 1

    for fnum in sorted(by_frame.keys()):
        cur_boxes = by_frame[fnum]
        unmatched = list(range(len(cur_boxes)))

        for t_idx, (tid, last_fn, last_cx, last_cy, t_recs) in enumerate(active_tracks):
            if fnum - last_fn > 3:
                continue
            best_dist = 0.04 # max 4% image width drift between frames
            best_m = -1
            for m in unmatched:
                cand = cur_boxes[m]
                dist = math.hypot(cand["bbox"][0] - last_cx, cand["bbox"][1] - last_cy)
                if dist < best_dist:
                    best_dist = dist
                    best_m = m
            if best_m != -1:
                rec = cur_boxes[best_m]
                t_recs.append(rec)
                active_tracks[t_idx] = (tid, fnum, rec["bbox"][0], rec["bbox"][1], t_recs)
                unmatched.remove(best_m)

        new_active = []
        for t in active_tracks:
            if fnum - t[1] > 3:
                finished_tracks.append(t[4])
            else:
                new_active.append(t)
        active_tracks = new_active

        for m in unmatched:
            rec = cur_boxes[m]
            new_active.append((next_tid, fnum, rec["bbox"][0], rec["bbox"][1], [rec]))
            next_tid += 1

    for t in active_tracks:
        finished_tracks.append(t[4])

    return finished_tracks

print("Testing nearest-distance tracking on scb_vid_3000...")
rec_3000 = [r for r in all_records if r["video_id"] == "scb_vid_3000"]
tracks_3000 = track_video_by_distance(rec_3000)
print(f"Total tracks in 3000: {len(tracks_3000)}")
long_tracks = [t for t in tracks_3000 if len(t) >= 16]
print(f"Tracks with >= 16 frames: {len(long_tracks)}")

seqs = 0
classes_cnt = Counter()
for t in long_tracks:
    runs = []
    cur_run = [t[0]]
    for i in range(len(t)-1):
        if t[i+1]["tempo_label"] == t[i]["tempo_label"]:
            cur_run.append(t[i+1])
        else:
            if len(cur_run) >= 16: runs.append(cur_run)
            cur_run = [t[i+1]]
    if len(cur_run) >= 16: runs.append(cur_run)

    for run in runs:
        for s in range(0, len(run)-16+1, 16):
            classes_cnt[run[s]["tempo_label"]] += 1
            seqs += 1

print(f"Non-overlapping (stride=16) sequences in scb_vid_3000: {seqs}, breakdown: {dict(classes_cnt)}")

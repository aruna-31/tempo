import sys; sys.path.append('scripts')
import test_scan_and_split
from collections import defaultdict, Counter

all_records, _, _ = test_scan_and_split.scan()

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

def track_video(records):
    # Group by frame_num
    by_frame = defaultdict(list)
    for r in records:
        by_frame[int(r["frame_num"])].append(r)

    active_tracks = [] # list of (track_id, last_fnum, last_box, list_of_records)
    finished_tracks = []
    next_tid = 1

    for fnum in sorted(by_frame.keys()):
        cur_boxes = by_frame[fnum]
        unmatched = list(range(len(cur_boxes)))
        
        # Match with active tracks where last_fnum is within 3 frames
        for t_idx, (tid, last_fn, last_box, t_recs) in enumerate(active_tracks):
            if fnum - last_fn > 3:
                continue
            best_iou = 0.3
            best_m = -1
            for m in unmatched:
                cand = cur_boxes[m]
                score = box_iou(last_box, cand["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_m = m
            if best_m != -1:
                matched_rec = cur_boxes[best_m]
                t_recs.append(matched_rec)
                active_tracks[t_idx] = (tid, fnum, matched_rec["bbox"], t_recs)
                unmatched.remove(best_m)

        # Still active vs stale
        new_active = []
        for t in active_tracks:
            if fnum - t[1] > 3:
                finished_tracks.append(t[3])
            else:
                new_active.append(t)
        active_tracks = new_active

        # Start new tracks for unmatched boxes
        for m in unmatched:
            rec = cur_boxes[m]
            new_active.append((next_tid, fnum, rec["bbox"], [rec]))
            next_tid += 1

    for t in active_tracks:
        finished_tracks.append(t[3])

    return finished_tracks

print("Testing IoU tracking on scb_vid_3000...")
rec_3000 = [r for r in all_records if r["video_id"] == "scb_vid_3000"]
tracks_3000 = track_video(rec_3000)
print(f"Generated {len(tracks_3000)} tracks in 3000")
long_tracks = [t for t in tracks_3000 if len(t) >= 16]
print(f"Tracks with >= 16 frames: {len(long_tracks)}")

# Check sequence generation with stride 16
seqs = 0
classes_cnt = Counter()
for t in long_tracks:
    # Check runs of pure behavior
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

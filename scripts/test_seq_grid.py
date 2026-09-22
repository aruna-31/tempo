import sys; sys.path.append('scripts')
import test_scan_and_split
from collections import defaultdict, Counter

all_records, _, _ = test_scan_and_split.scan()

# Group by student_track_id as in build_clean_dataset.py
# bx=0.07, by=0.14
tracks = defaultdict(list)
for r in all_records:
    tid = f"{r['video_id']}_s{int(round(r['bbox'][0]/0.07))}_{int(round(r['bbox'][1]/0.14))}"
    tracks[tid].append(r)

SEQ_LEN = 16

for stride_val in [16, 8, 4]:
    for max_gap in [3, 5, 8, 10]:
        seq_counts = Counter()
        for track_id, items in tracks.items():
            if len(items) < SEQ_LEN:
                continue
            items.sort(key=lambda x: int(x["frame_num"]))

            for s in range(0, len(items) - SEQ_LEN + 1, stride_val):
                window = items[s : s + SEQ_LEN]
                frame_nums = [int(w["frame_num"]) for w in window]
                is_continuous = all(
                    1 <= frame_nums[i + 1] - frame_nums[i] <= max_gap
                    for i in range(len(frame_nums) - 1)
                )
                if not is_continuous:
                    continue
                labels = [w["tempo_label"] for w in window]
                if len(set(labels)) != 1:
                    continue
                seq_counts[labels[0]] += 1
        print(f"stride={stride_val:>2d}, max_gap={max_gap:>2d} -> Total: {sum(seq_counts.values()):>5d} | {dict(seq_counts)}")

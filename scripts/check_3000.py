import sys; sys.path.append('scripts')
import test_scan_and_split
all_records, _, _ = test_scan_and_split.scan()
rec = [r for r in all_records if r['video_id'] == 'scb_vid_3000']
tracks = {}
for r in rec:
    tid = f"{r['video_id']}_s{int(round(r['bbox'][0]/0.07))}_{int(round(r['bbox'][1]/0.14))}"
    tracks.setdefault(tid, []).append(r)

print('Total tracks in scb_vid_3000:', len(tracks))
for tid, items in list(tracks.items())[:10]:
    items.sort(key=lambda x: int(x['frame_num']))
    fnums = [int(it['frame_num']) for it in items]
    lbls = [it['tempo_label'] for it in items]
    print(tid, 'len:', len(items), 'fnums:', fnums[:10], 'lbls:', set(lbls))

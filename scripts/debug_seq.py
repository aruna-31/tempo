import test_scan_and_split
all_records, _, _ = test_scan_and_split.scan()
rec_0006 = [r for r in all_records if r["video_id"] == "scb_vid_0006"]
print("Total records for scb_vid_0006:", len(rec_0006))
tracks = {}
for r in rec_0006:
    tid = f"{r['video_id']}_s{int(round(r['bbox'][0]/0.07))}_{int(round(r['bbox'][1]/0.14))}"
    tracks.setdefault(tid, []).append(r)

print("Unique tracks in 0006:", len(tracks))
for tid, items in list(tracks.items())[:10]:
    items.sort(key=lambda x: int(x["frame_num"]))
    fnums = [int(it["frame_num"]) for it in items]
    lbls = [it["tempo_label"] for it in items]
    print(tid, "len:", len(items), "fnums:", fnums[:10], "lbls:", set(lbls))

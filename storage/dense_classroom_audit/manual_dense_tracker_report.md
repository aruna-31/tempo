# Dense tracker manual benchmark

{
  "video": "C:\\Users\\aruna\\Downloads\\classroom.mp4",
  "iou_threshold": 0.5,
  "manual_annotation_note": "Five fixed frames were manually boxed; boxes cover clearly visible heads/upper bodies and partial edge people.",
  "benchmark_frames": [
    0,
    86,
    171,
    256,
    342
  ],
  "per_frame": {
    "0": {
      "manual_students": 28,
      "detections": 31,
      "active_tracks": 0,
      "tp": 19,
      "fp": 12,
      "fn": 9,
      "precision": 0.6129032258064516,
      "recall": 0.6785714285714286,
      "f1": 0.6440677966101694
    },
    "86": {
      "manual_students": 26,
      "detections": 27,
      "active_tracks": 23,
      "tp": 16,
      "fp": 11,
      "fn": 10,
      "precision": 0.5925925925925926,
      "recall": 0.6153846153846154,
      "f1": 0.6037735849056604
    },
    "171": {
      "manual_students": 27,
      "detections": 29,
      "active_tracks": 26,
      "tp": 15,
      "fp": 14,
      "fn": 12,
      "precision": 0.5172413793103449,
      "recall": 0.5555555555555556,
      "f1": 0.5357142857142857
    },
    "256": {
      "manual_students": 26,
      "detections": 28,
      "active_tracks": 26,
      "tp": 17,
      "fp": 11,
      "fn": 9,
      "precision": 0.6071428571428571,
      "recall": 0.6538461538461539,
      "f1": 0.6296296296296297
    },
    "342": {
      "manual_students": 23,
      "detections": 26,
      "active_tracks": 24,
      "tp": 15,
      "fp": 11,
      "fn": 8,
      "precision": 0.5769230769230769,
      "recall": 0.6521739130434783,
      "f1": 0.6122448979591837
    }
  },
  "aggregate_detection": {
    "tp": 82,
    "fp": 59,
    "fn": 48,
    "precision": 0.5815602836879432,
    "recall": 0.6307692307692307,
    "f1": 0.6051660516605165,
    "student_coverage": 0.6307692307692307
  },
  "tracking": {
    "unique_observed_tracks": 63,
    "peak_active_tracks": 26,
    "mean_active_tracks_on_benchmark_frames": 19.8,
    "track_fragmentation_proxy": 11,
    "track_reappearances_proxy": 11,
    "id_switches": "not measurable from anonymous five-frame boxes; no persistent manual identity labels were assigned"
  },
  "artifacts": [
    "manual_benchmark_frame_0000.jpg",
    "manual_benchmark_frame_0086.jpg",
    "manual_benchmark_frame_0171.jpg",
    "manual_benchmark_frame_0256.jpg",
    "manual_benchmark_frame_0342.jpg",
    "appearance_dense_tracker_benchmark.mp4"
  ]
}
